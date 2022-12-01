#!/usr/bin/python

import os
from glob import glob

import numpy as np
import tensorflow as tf
from dataset import Dataset
from network import Unet, SpatialConfigurationNet, UnetAvgLinear3D
from tensorflow.keras import mixed_precision
from tqdm import tqdm

import utils.io.image
import utils.sitk_image
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from utils.segmentation.segmentation_test import SegmentationTest
from bin.semi_supervised_learning.mmwhs.testset_evaluation.reorient_prediction_to_reference import perform_reorientation


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MainLoop(MainLoopBase):
    def __init__(self,
                 network,
                 unet,
                 normalized_prediction,
                 network_parameters,
                 load_model_filename,
                 iter,
                 output_folder_name=''):
        super().__init__()
        self.use_mixed_precision = True
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        self.num_labels = 8
        self.data_format = 'channels_first'
        self.channel_axis = 1
        self.network = network
        self.unet = unet
        self.normalized_prediction = normalized_prediction
        self.padding = 'same'
        self.output_folder_name = output_folder_name
        self.current_iter = iter

        self.output_background_local = False
        self.input_background_spatial = False
        self.output_background_spatial = False
        self.output_background_final = True

        self.base_output_folder = '/media0/franz/experiments/semi_supervised_learning/mmwhs17'
        self.local_base_folder = '/media0/franz/datasets/heart/mmwhs17'
        # self.image_base_folder = '/media1/datasets/segmentation/ilearnheart/images/'
        # self.image_id_list = glob(os.path.join(self.image_base_folder, '*.nii.gz'))
        # self.image_id_list = [os.path.split(i)[-1] for i in self.image_id_list]
        self.load_model_filename = load_model_filename

        self.image_size = [96] * 3
        # if modality == 'ct':
        #     self.image_extend = [192] * 3
        # else:
        #     self.image_extend = [192] * 3
        #     #self.image_extend = [256] * 3
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
                                       cv=None,
                                       modality=modality,
                                       setup_folder_to_use='setup',
                                       cached_datasource=False,
                                       data_format=self.data_format,
                                       image_pixel_type=np.float16 if mixed_precision else np.float32,
                                       save_debug_images=False)


    def init_model(self):
        self.norm_moving_average = tf.Variable(10.0)
        self.model = self.network(**self.network_parameters)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name=self.model.name, additional_info=self.output_folder_name, use_timestamp=False)

    def init_datasets(self):
        dataset = Dataset(**self.dataset_parameters)
        self.dataset_inference = dataset.dataset_inference()

    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        labels = list(range(self.num_labels))
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator='linear',
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)

        # for image_id in tqdm(self.image_id_list, desc='Testing'):
        #     dataset_entry = self.dataset_test.get({'image_id': image_id})
        num_entries = self.dataset_inference.num_entries()
        for _ in tqdm(range(num_entries), desc='Testing'):
            dataset_entry = self.dataset_inference.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            generators = dataset_entry['generators']
            transformations = dataset_entry['transformations']
            prediction, local_prediction, spatial_prediction, local_prediction_wo_sigmoid, spatial_prediction_wo_sigmoid = self.model(np.expand_dims(generators['image'], axis=0), False)

            prediction = np.squeeze(prediction, axis=0)
            local_prediction = np.squeeze(local_prediction, axis=0)
            spatial_prediction = np.squeeze(spatial_prediction, axis=0)
            input = datasources['image']
            transformation = transformations['image']

            prediction_labels = segmentation_test.get_label_image(prediction, input, self.image_spacing, transformation)
            origin = transformation.TransformPoint(np.zeros(3, np.float64))

            utils.io.image.write(prediction_labels, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '.mha'))
            # utils.io.image.write_multichannel_np(prediction, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            # utils.io.image.write_multichannel_np(local_prediction, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_local_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
            # utils.io.image.write_multichannel_np(spatial_prediction, self.output_folder_handler.path_for_iteration(self.current_iter, current_id + '_spatial_prediction.mha'), output_normalization_mode=(0, 1), data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)


if __name__ == '__main__':

    experiment_base = '/media0/franz/experiments/semi_supervised_learning/mmwhs17/unet'
    # experiment_dict = {
    #     's5':  'supervised/ct/cv1/setup_supervised_s5/2021-02-17_10-08-58',
    #     's7':  'supervised/ct/cv1/setup_supervised_s7/2021-02-16_18-55-10',
    #     's14': 'supervised/ct/cv1/setup_supervised_s14/2021-02-17_10-08-51',
    #     's20': 'supervised/ct/cv1/setup_supervised_s20/2021-02-16_18-58-06',
    # }

    experiment_dict = {
        's7':  'supervised/mr/cv1/setup_supervised_s7/EXPERIMENTS-2021-02-18/2021-02-18_17-13-07',
    }

    modality = 'mr'
    iter = 20000
    do_reorientation = True


    for exp_name in experiment_dict.values():
        load_model_filename = os.path.join(experiment_base, exp_name, 'weights', f'ckpt-{iter}')
        output_folder_name = os.path.join('inference', exp_name)

        create_dir(output_folder_name)


        network_parameters = {'local_network_parameters': {'num_filters_base': 128, 'num_levels': 5, 'dropout_ratio': 0.1},
                              'spatial_network_parameters': {'num_filters_base': 32, 'num_levels': 4, 'dropout_ratio': 0.1},
                              'final_network_parameters': {'num_filters_base': 32, 'num_levels': 4, 'dropout_ratio': 0.1},
                              'spatial_downsample': 4,
                              'activation': 'lrelu',
                              'local_channel_dropout_ratio': 0.25,
                              'local_activation': 'sigmoid',
                              'spatial_activation': 'sigmoid'}

        loop = MainLoop(Unet,
                        UnetAvgLinear3D,
                        False,
                        network_parameters,
                        load_model_filename,
                        iter,
                        output_folder_name=output_folder_name)
        loop.run_test()

        if do_reorientation:
            reference_folder = '/media0/franz/datasets/heart/mmwhs17/all_test'
            perform_reorientation(os.path.join(experiment_base, output_folder_name, f'iter_{iter}'), reference_folder, modality)




