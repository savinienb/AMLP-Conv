

import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import conv2d, concat_channels, upsample2d, max_pool2d #, deform_conv_2d
from tensorflow_train.layers.invariant_layers import conv2d_bilinear_scale_invariant
from tensorflow_train.networks.unet import UnetParallel2D
from tensorflow_train.networks.unet_lstm import UnetLstm2D
from tensorflow_train.networks.downsampling_net import DownsamplingNet2D

class UnetParallelScaleInvariant2D(UnetParallel2D):
    def conv(self, node, current_level, postfix, is_training):
        return conv2d_bilinear_scale_invariant(node,
                                      self.num_filters(current_level),
                                      [3, 3],
                                      name='conv' + str(current_level) + postfix,
                                      activation=self.activation,
                                      use_batch_norm=self.use_batch_norm,
                                      is_training=is_training,
                                      data_format=self.data_format)

class UnetLstmScaleInvariant2D(UnetLstm2D):
    def conv(self, node, current_level, postfix, is_training):
        return conv2d_bilinear_scale_invariant(node,
                                      self.num_filters(current_level),
                                      [3, 3],
                                      name='conv' + str(current_level) + postfix,
                                      activation=self.activation,
                                      use_batch_norm=self.use_batch_norm,
                                      is_training=is_training,
                                      data_format=self.data_format)

class DownsamplingNetScaleInvariant2D(DownsamplingNet2D):
    def conv(self, node, current_level, postfix, is_training):
        return conv2d_bilinear_scale_invariant(node,
                                      self.num_filters(current_level),
                                      [3, 3],
                                      name='conv' + str(current_level) + postfix,
                                      activation=self.activation,
                                      use_batch_norm=self.use_batch_norm,
                                      is_training=is_training,
                                      data_format=self.data_format)

class PoseTrackingNet(object):
    def __init__(self,
                 num_heatmaps,
                 heatmap_activation=None,
                 heatmap_kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                 use_batch_norm=False,
                 kernel_size=None,
                 activation=tf.nn.relu,
                 data_format='channels_first'):
        self.num_heatmaps = num_heatmaps
        self.heatmap_activation = heatmap_activation
        self.heatmap_kernel_initializer = heatmap_kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.kernel_size = kernel_size
        if self.kernel_size is None:
            self.kernel_size = [3, 3]
        self.activation = activation
        self.data_format = data_format
        self.lstm_cells = []
        self.num_filters_resample_net = 128
        self.num_levels_resample_net = 3
        self.num_filters_head_net = 128
        self.num_levels_head_net = 4
        self.num_filters_pose_net = 128
        self.num_levels_pose_net = 4
        self.downsampling_net = DownsamplingNetScaleInvariant2D(num_filters_base=self.num_filters_resample_net,
                                                                num_levels=self.num_levels_resample_net,
                                                                activation=self.activation,
                                                                use_batch_norm=self.use_batch_norm,
                                                                data_format=self.data_format)
        self.unet_head = UnetParallelScaleInvariant2D(num_filters_base=self.num_filters_head_net,
                                                      num_levels=self.num_levels_head_net,
                                                      activation=self.activation,
                                                      use_batch_norm=self.use_batch_norm,
                                                      data_format=self.data_format)
        self.unet_lstm = UnetLstmScaleInvariant2D(num_filters_base=self.num_filters_pose_net,
                                                  num_levels=self.num_levels_pose_net,
                                                  activation=self.activation,
                                                  use_batch_norm=self.use_batch_norm,
                                                  data_format=self.data_format)

    def heatmap_conv(self, node, num_filters, name, is_training):
        return conv2d(node,
                      num_filters,
                      self.kernel_size,
                      name=name,
                      activation=self.heatmap_activation,
                      kernel_initializer=self.heatmap_kernel_initializer,
                      use_batch_norm=False,
                      is_training=is_training,
                      data_format=self.data_format)

    def resample_net(self, data, is_training, reuse_scope=False):
        with tf.variable_scope('data_conv', reuse=reuse_scope):
            data_conv = self.downsampling_net(data, is_training)
            return data_conv

    def head_net(self, data_conv, is_training, reuse_scope=False):
        with tf.variable_scope('head_net', reuse=reuse_scope):
            unet_head = self.unet_head(data_conv, is_training)
            heatmap_head = self.heatmap_conv(unet_head, 1, name='heatmap_head', is_training=is_training)
            return heatmap_head

    def head_contracting_net(self, data_conv, heatmap_head, heatmap_head_mask, is_training, reuse_scope=False):
        with tf.variable_scope('head_contracting', reuse=reuse_scope):
            heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask, clip_value_max=1.0, clip_value_min=0.0, name='heatmap_head_mask_reshape_clamped')
            heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
            heatmap_single_head_and_image = concat_channels([heatmap_single_head, data_conv], name='heatmap_single_head_and_image', data_format=self.data_format)
            #heatmap_single_head_and_image = conv2d(heatmap_single_head_and_image, self.num_features_per_level[0], self.kernel_size, name='conv0', activation=tf.nn.relu, weight_filler='he_normal', stddev=None, use_batch_norm=self.use_batch_norm, is_training=is_training)
            head_contracting = self.unet_lstm.contracting(heatmap_single_head_and_image, is_training)
            return head_contracting

    def data_conv_contracting_net(self, data_conv, is_training, reuse_scope=False):
        with tf.variable_scope('data_conv_contracting', reuse=reuse_scope):
            #data_conv = conv2d(data_conv, self.num_features_per_level[0], self.kernel_size, name='conv0', activation=tf.nn.relu, weight_filler='he_normal', stddev=None, use_batch_norm=self.use_batch_norm, is_training=is_training)
            head_contracting = self.unet_lstm.contracting(data_conv, is_training)
            return head_contracting

    def heatmaps_single_person_lstm_expanding_net(self, unet_level_nodes, lstm_input_states, is_training, reuse_scope=False):
        with tf.variable_scope('lstm_expanding', reuse=reuse_scope):
            single_net = self.unet_lstm.parallel_and_expanding_with_input_states(unet_level_nodes, lstm_input_states, is_training)
            heatmaps_single_person = self.heatmap_conv(single_net, self.num_heatmaps, name='heatmaps_single_person', is_training=is_training)
            return heatmaps_single_person, self.unet_lstm.lstm_input_states, self.unet_lstm.lstm_output_states

    def training_net(self, data_volume, heatmap_head_mask, is_training):
        print('Creating training net...')
        data_shape = data_volume.get_shape().as_list()
        print(data_shape)
        if self.data_format == 'channels_first':
            frame_stack_axis = 2
        else:
            frame_stack_axis = 1

        num_frames = data_shape[frame_stack_axis]
        data_conv_list = []
        heatmaps_single_person_list = []

        first = True
        for i in range(num_frames):
            if self.data_format == 'channels_first':
                current_data_conv = self.resample_net(data_volume[:, :, i, :, :], is_training, not first)
            else:
                current_data_conv = self.resample_net(data_volume[:, i, :, :, :], is_training, not first)
            data_conv_list.append(current_data_conv)
            first = False

        #with tf.variable_scope('first_frame_net'):
        current_data_conv = data_conv_list[0]
        heatmap_head = self.head_net(current_data_conv, is_training, False)
        head_contracting = self.head_contracting_net(current_data_conv, heatmap_head, heatmap_head_mask, is_training, False)

        heatmaps_single_person, _, lstm_output_states = self.heatmaps_single_person_lstm_expanding_net(head_contracting, None, is_training, False)
        heatmaps_single_person_list.append(heatmaps_single_person)

        first = True
        for i in range(1, num_frames):
            current_data_conv = data_conv_list[i]
            data_conv_contracting = self.data_conv_contracting_net(current_data_conv, is_training, not first)
            heatmaps_single_person, _, lstm_output_states = self.heatmaps_single_person_lstm_expanding_net(data_conv_contracting, lstm_output_states, is_training, True)
            heatmaps_single_person_list.append(heatmaps_single_person)
            first = False

        data_conv_concat = tf.stack(data_conv_list, axis=frame_stack_axis, name='data_conv_concat')
        heatmaps_single_person_concat = tf.stack(heatmaps_single_person_list, axis=frame_stack_axis, name='heatmaps_single_person_concat')

        return data_conv_concat, heatmap_head, heatmaps_single_person_concat
