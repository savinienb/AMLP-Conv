#!/usr/bin/python

from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision
import tensorflow_probability as tfp

import utils.io.image
# from tensorflow_train_v2.dataset.dataset_iterator_multiprocessing import DatasetIteratorMultiprocessing as DatasetIterator
# from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator as DatasetIterator
from tensorflow_train_v2.losses.semantic_segmentation_losses import generalized_dice_loss, sigmoid_cross_entropy_with_logits
from tensorflow_train_v2.train_loop import MainLoopBase
import utils.sitk_image
from tensorflow_train_v2.utils.loss_metric_logger import LossMetricLogger
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from utils.segmentation.segmentation_test import SegmentationTest
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from utils.segmentation.metrics import DiceMetric, SurfaceDistanceMetric
from datasets.pyro_dataset import PyroClientDataset
from network import Unet, SpatialConfigurationNet, UnetAvgLinear3D, UnetMixer, UnetMixerAvgLinear3D, AMLPLine
import os
import socket


class MainLoop(MainLoopBase):
    def __init__(self,
                 modality,
                 cv,
                 network,
                 unet,
                 normalized_prediction,
                 loss,
                 local_loss,
                 network_parameters,
                 learning_rate,
                 lr_decay,
                 loss_factor_local=0.0,
                 loss_factor_spatial=0.0,
                 image_size_per_dim=96,
                 setup_folder_to_use='',
                 cached_datasource=True,
                 dataset_threads=4,
                 output_background_local=False,
                 input_background_spatial=False,
                 output_background_spatial=False,
                 output_background_final=True,
                 output_folder_name=''):
        super().__init__()

        self.use_mixed_precision = False
        hostname = socket.gethostname()
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else None

        self.use_mixed_precision = False
        if hostname == 'rvlab-gr':
            self.use_mixed_precision = False
        if hostname == 'Tesla':
            if cuda_visible_devices is not None and cuda_visible_devices != '0':
                self.use_mixed_precision = False

        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

        self.host_id = f'{hostname}_gpu:{cuda_visible_devices}'


        self.cv = cv
        self.batch_size = 1
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.max_iter = 60000
        self.test_iter = 10000  # 1000
        self.snapshot_iter = self.test_iter
        if cv == 0:
            self.test_iter = 20000
            self.snapshot_iter = 10000
        self.disp_iter = 100
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.000001
        self.data_format = 'channels_first'
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.normalized_prediction = normalized_prediction
        self.padding = 'same'
        self.output_folder_name = output_folder_name

        self.output_background_local = output_background_local
        self.input_background_spatial = input_background_spatial
        self.output_background_spatial = output_background_spatial
        self.output_background_final = output_background_final

        self.loss_factor_local = loss_factor_local
        self.loss_factor_spatial = loss_factor_spatial
        self.setup_folder_to_use = setup_folder_to_use
        self.dataset_threads = dataset_threads
        self.use_pyro_dataset = False
        self.has_validation_groundtruth = cv != 0

        hostname = socket.gethostname()


        self.base_output_folder = '/home/sav/experiments/mmwhs17'
        self.local_base_folder = '/mnt/ssddata/datasets/mmwhs17'
        #self.wandb_folder = self.base_output_folder



        self.num_labels = 8

        '''
        self.image_size = [64]*3
        if modality == 'ct':
            #3
            self.image_spacing = [3/96*self.image_size[0]*0.75]*3
            self.image_spacing = [3]*3
            #self.image_spacing = [3/96*64*0.75]*3
        else:
            self.image_spacing = [4, 4, 4]
        self.input_gaussian_sigma = 1.0
        self.label_gaussian_sigma = 1.0
        '''
        # self.image_size = [96] * 3
        # self.image_size = [64] * 3
        self.image_size = [image_size_per_dim] * 3
        self.image_extend = [192] * 3

        self.image_spacing = [extend / size for extend, size in zip(self.image_extend, self.image_size)]

        self.network_parameters = dict(num_labels=self.num_labels,
                                       actual_network=self.unet,
                                       padding=self.padding,
                                       data_format=self.data_format,
                                       **network_parameters)

        self.dataset_parameters = dict(base_folder=self.local_base_folder,
                                       image_size=list(reversed(self.image_size)),
                                       image_spacing=list(reversed(self.image_spacing)),
                                       cv=self.cv,
                                       modality=modality,
                                       setup_folder_to_use=self.setup_folder_to_use,
                                       cached_datasource=cached_datasource,
                                       data_format=self.data_format,
                                       image_pixel_type=np.float16 if mixed_precision else np.float32,
                                       save_debug_images=False)

        # self.metric_names = OrderedDict([(name, ['mean_{}'.format(name)] + list(map(lambda x: '{}_{}'.format(name, x), range(1, self.num_labels)))) for name in ['dice', 'sd_mean', 'sd_median', 'sd_std', 'sd_max']])
        self.metric_names = OrderedDict([(name, ['mean_{}'.format(name)] + list(map(lambda x: '{}_{}'.format(name, x), range(1, self.num_labels)))) for name in ['dice']])

        self.loss_function = loss
        self.local_loss_function = local_loss

    def run(self):
        super(MainLoop, self).run()

    def init_all(self):
        """
        Init all objects. Calls abstract init_*() functions.
        """
        # super(MainLoop, self).init_all()
        self.init_model()
        self.init_optimizer()
        self.init_output_folder_handler()
        self.init_checkpoint()
        self.init_checkpoint_manager()
        self.init_datasets()
        self.init_loggers()


    def init_model(self):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)  # original 0.999  # testing 0.99
        self.norm_moving_average = tf.Variable(10.0)
        self.model = self.network(**self.network_parameters)

    def save_model(self):
        """
        Save the model.
        """
        old_weights = [tf.keras.backend.get_value(var) for var in self.model.trainable_variables]
        new_weights = [tf.keras.backend.get_value(self.ema.average(var)) for var in self.model.trainable_variables]
        for var, weights in zip(self.model.trainable_variables, new_weights):
            tf.keras.backend.set_value(var, weights)
        super(MainLoop, self).save_model()
        for var, weights in zip(self.model.trainable_variables, old_weights):
            tf.keras.backend.set_value(var, weights)

    def init_optimizer(self):
        if self.lr_decay:
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale=tf.mixed_precision.experimental.DynamicLossScale(initial_loss_scale=2 ** 15, increment_period=1000))

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name=self.model.name, additional_info=self.output_folder_name)

    def init_datasets(self):
        network_image_size = self.image_size
        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('labels', [1] + network_image_size),
                                                  ])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('labels', network_image_size + [1]),
                                                  ])

        data_generator_types = {'image': tf.float16 if self.use_mixed_precision else tf.float32,
                                'labels': tf.uint8,
                                }


        dataset = Dataset(**self.dataset_parameters)
        if self.use_pyro_dataset:
            from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator 

            uri_base = 'PYRO:mmhws_dataset'
            uri = f'{uri_base}@{self.server_hostname}:{self.server_port_train}'
            print('using pyro uri', uri)

            self.dataset_train = PyroClientDataset(uri, compression_type='lz4', **self.dataset_parameters)
            self.dataset_train_iter = DatasetIterator(dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size)

        else:
            from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator

            self.dataset_train = dataset.dataset_train()
            self.dataset_train_iter = DatasetIterator(dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size)

        self.dataset_val = dataset.dataset_val()


    def init_loggers(self):
        self.loss_metric_logger_train = LossMetricLogger('train',
                                                         self.output_folder_handler.path('train'),
                                                         self.output_folder_handler.path('train.csv'))
        self.loss_metric_logger_val = LossMetricLogger('test',
                                                       self.output_folder_handler.path('test'),
                                                       self.output_folder_handler.path('test.csv'))

    def split_labels_tf(self, labels, w_batch_dim):
        if w_batch_dim:
            axis = self.channel_axis
        else:
            # axis wo batch dimension
            axis = 0 if self.data_format == 'channels_first' else -1
        split_labels = tf.one_hot(tf.squeeze(labels, axis=axis), depth=self.num_labels, axis=axis)
        return split_labels

    @tf.function
    def call_model_and_loss(self, image, labels, training):
        prediction, local_prediction, spatial_prediction, local_prediction_wo_sigmoid, spatial_prediction_wo_sigmoid = self.model(image, training=training)
        losses = self.losses(labels, prediction, local_prediction_wo_sigmoid, spatial_prediction_wo_sigmoid)
        return (prediction, local_prediction, spatial_prediction, local_prediction_wo_sigmoid, spatial_prediction_wo_sigmoid), losses

    @tf.function
    def train_step(self):
        image, labels = self.dataset_train_iter.get_next()
        labels = self.split_labels_tf(labels, w_batch_dim=True)
        with tf.GradientTape() as tape:
            _, losses = self.call_model_and_loss(image, labels, training=True)
            if self.reg_constant > 0:
                losses['loss_reg'] = self.reg_constant * tf.reduce_sum(self.model.losses)
            loss = tf.reduce_sum(list(losses.values()))
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        variables = self.model.trainable_weights
        metric_dict = losses
        clip_norm = self.norm_moving_average * 5
        if self.use_mixed_precision:
            scaled_grads = tape.gradient(scaled_loss, variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            loss_scale = self.optimizer.loss_scale
            metric_dict.update({'loss_scale': loss_scale})
        else:
            grads = tape.gradient(loss, variables)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
        if tf.math.is_finite(norm):
            alpha = 0.01
            self.norm_moving_average.assign(alpha * tf.minimum(norm, clip_norm) + (1 - alpha) * self.norm_moving_average)
        metric_dict.update({'norm': norm, 'norm_average': self.norm_moving_average})
        self.optimizer.apply_gradients(zip(grads, variables))
        self.ema.apply(variables)

        self.loss_metric_logger_train.update_metrics(metric_dict)

    @tf.function
    def losses(self, mask, prediction, local_prediction, spatial_prediction):
        mask_wo_background = mask[:, 1:, :, :, :]
        if self.normalized_prediction:
            loss_total = self.loss_function(labels=mask if self.output_background_final else mask_wo_background, logits_as_probability=prediction, data_format=self.data_format)
        else:
            loss_total = self.loss_function(labels=mask if self.output_background_final else mask_wo_background, logits=prediction, data_format=self.data_format)
        losses_dict = {'loss': loss_total}

        if self.loss_factor_local > 0:
            loss_local = self.local_loss_function(labels=mask if self.output_background_local else mask_wo_background, logits=local_prediction)
            losses_dict['loss_local'] = loss_local
        if self.loss_factor_spatial > 0:
            loss_spatial = self.local_loss_function(labels=mask if self.output_background_spatial else mask_wo_background, logits=spatial_prediction)
            losses_dict['loss_spatial'] = loss_spatial

        return losses_dict

    def get_summary_dict(self, segmentation_statistics, name):
        mean_list = segmentation_statistics.get_metric_mean_list(name)
        mean_of_mean_list = np.mean(mean_list)
        return OrderedDict(list(zip(self.metric_names[name], [mean_of_mean_list] + mean_list)))

    def test(self):
        print('Testing...')

        if self.current_iter != 0:
            old_weights = [tf.keras.backend.get_value(var) for var in self.model.trainable_variables]
            new_weights = [tf.keras.backend.get_value(self.ema.average(var)) for var in self.model.trainable_variables]
            for var, weights in zip(self.model.trainable_variables, new_weights):
                tf.keras.backend.set_value(var, weights)

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        labels = list(range(self.num_labels))
        labels_wo_background = list(range(1, self.num_labels))
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator='linear',
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)
        if self.cv != 0:
            segmentation_statistics = SegmentationStatistics(labels_wo_background,
                                                         self.output_folder_handler.path_for_iteration(self.current_iter),
                                                         metrics=OrderedDict([('dice', DiceMetric()),
                                                                              # (('sd_mean', 'sd_median', 'sd_std', 'sd_max'), SurfaceDistanceMetric())
                                                                              ]))
        num_entries = self.dataset_val.num_entries()
        for _ in tqdm(range(num_entries), desc='Testing'):
            dataset_entry = self.dataset_val.get_next()

            if self.cv != 0:
                dataset_entry['generators']['labels'] = self.split_labels_tf(dataset_entry['generators']['labels'], w_batch_dim=False)
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            generators = dataset_entry['generators']
            transformations = dataset_entry['transformations']
            if self.has_validation_groundtruth:
                (prediction, local_prediction, spatial_prediction, local_prediction_wo_sigmoid, spatial_prediction_wo_sigmoid), losses = self.call_model_and_loss(np.expand_dims(generators['image'], axis=0), np.expand_dims(generators['labels'], axis=0), False)
            else:
                prediction, local_prediction, spatial_prediction, local_prediction_wo_sigmoid, spatial_prediction_wo_sigmoid = self.model(np.expand_dims(generators['image'], axis=0), False)

            prediction = np.squeeze(tf.nn.softmax(prediction, axis=1 if self.data_format == 'channels_first' else -1), axis=0)
            local_prediction = np.squeeze(local_prediction, axis=0)
            spatial_prediction = np.squeeze(spatial_prediction, axis=0)
            input = datasources['image']
            transformation = transformations['image']

            prediction_labels = segmentation_test.get_label_image(prediction, input, self.image_spacing, transformation)
            # utils_local.io.image.write(prediction_labels, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '.mha'))
            origin = transformation.TransformPoint(np.zeros(3, np.float64))

            # utils_local.io.image.write(datasources['image'], self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_source.mha'))
            # utils_local.io.image.write_np(generators['image'], self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_input.mha'))
            utils.io.image.write(prediction_labels, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '.mha'))
            utils.io.image.write_multichannel_np(prediction, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            # utils.io.image.write_multichannel_np(local_prediction, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_local_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            # utils.io.image.write_multichannel_np(spatial_prediction, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_spatial_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)

            if self.has_validation_groundtruth:
                groundtruth = datasources['labels']
                segmentation_statistics.add_labels(current_id, prediction_labels, groundtruth)

        if self.has_validation_groundtruth:
            segmentation_statistics.finalize()
            summary_values = OrderedDict()
            for name in self.metric_names.keys():
                summary_values.update(self.get_summary_dict(segmentation_statistics, name))
            self.loss_metric_logger_val.update_metrics(summary_values)
        if self.cv != 0:
            self.loss_metric_logger_val.finalize(self.current_iter)

        if self.current_iter != 0:
            for var, weights in zip(self.model.trainable_variables, old_weights):
                tf.keras.backend.set_value(var, weights)

if __name__ == '__main__':

    
    network_parameters = {}

    image_size_per_dim = 64
    setup_folder_to_use = 'setup_supervised_s14'


    cached_datasource = True
    dataset_threads = 4

    lr_decay = False
    modalities = ['mr','ct']

    for modality in modalities:
        if modality is 'mr':
            from dataset_mri import Dataset
        else:
            from dataset_ct import Dataset
    # for modality in ['c']:
        for i in [1, 2, 3]:
            for _ in range(1):

                output_folder_name = f'{modality}/cv{i}/{setup_folder_to_use}/{image_size_per_dim}/'


                loop = MainLoop(modality,
                                i,
                                UnetMixer,
                                UnetMixerAvgLinear3D,
                                False,
                                generalized_dice_loss,
                                sigmoid_cross_entropy_with_logits,
                                network_parameters,
                                0.0001,
                                lr_decay,
                                loss_factor_local=0.0,
                                loss_factor_spatial=0.0,
                                image_size_per_dim=image_size_per_dim,
                                setup_folder_to_use=setup_folder_to_use,
                                cached_datasource=cached_datasource,
                                dataset_threads=dataset_threads,
                                output_folder_name=output_folder_name)
                loop.run()
