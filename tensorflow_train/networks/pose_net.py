
import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import conv2d, concat_channels, upsample2d, max_pool2d #, deform_conv_2d
from tensorflow_train.layers.invariant_layers import conv2d_scale_invariant
from tensorflow_train.networks.unet import UnetClassic2D
from tensorflow_train.networks.downsampling_net import DownsamplingNetBase2D

class DownsamplingNet2Conv2D(DownsamplingNetBase2D):
    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '_0', is_training)
        node = self.conv(node, current_level, '_1', is_training)
        return node

class MultiPersonPoseScnet(object):
    def __init__(self,
                 num_heatmaps,
                 heatmap_activation=tf.nn.tanh,
                 heatmap_kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
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
        self.num_filters_local_net = 128
        self.num_levels_local_net = 5
        self.num_filters_spatial_net = 128
        self.num_levels_spatial_net = 5

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
            downsampling_net = DownsamplingNet2Conv2D(num_filters_base=self.num_filters_resample_net,
                                                 num_levels=self.num_levels_resample_net,
                                                 activation=self.activation,
                                                 use_batch_norm=self.use_batch_norm,
                                                 data_format=self.data_format)
            data_conv = downsampling_net(data, is_training)
            return data_conv

    def local_net(self, intput, is_training, reuse_scope=False):
        with tf.variable_scope('local_net', reuse=reuse_scope):
            unet = UnetClassic2D(num_filters_base=self.num_filters_local_net,
                                  num_levels=self.num_levels_local_net,
                                  activation=self.activation,
                                  use_batch_norm=self.use_batch_norm,
                                  data_format=self.data_format)
            unet_head = unet(intput, is_training)
            heatmap_head = self.heatmap_conv(unet_head, 1, name='heatmap_head', is_training=is_training)
            heatmaps_local = self.heatmap_conv(unet_head, self.num_heatmaps, name='heatmaps_local', is_training=is_training)
            return heatmap_head, heatmaps_local

    def spatial_net(self, input, is_training, reuse_scope=False):
        with tf.variable_scope('spatial_net', reuse=reuse_scope):
            unet = UnetClassic2D(num_filters_base=self.num_filters_local_net,
                                  num_levels=self.num_levels_local_net,
                                  activation=self.activation,
                                  use_batch_norm=self.use_batch_norm,
                                  data_format=self.data_format)
            unet_head = unet(input, is_training)
            heatmaps_spatial = self.heatmap_conv(unet_head, self.num_heatmaps, name='heatmaps_spatial', is_training=is_training)
            return heatmaps_spatial

    def __call__(self, data, heatmap_head_mask, is_training):
        data_conv = self.resample_net(data, is_training)

        heatmap_head, heatmaps_local = self.local_net(data_conv, is_training)
        heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask, clip_value_max=1.0, clip_value_min=0.0, name='heatmap_head_mask_reshape_clamped')
        heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
        heatmaps_local_with_single_head_and_image = concat_channels([heatmaps_local, heatmap_single_head, data_conv], name='heatmaps_local_with_single_head_and_image', data_format=self.data_format)

        heatmaps_spatial_single_person = self.spatial_net(heatmaps_local_with_single_head_and_image, is_training)
        heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

        return data_conv, heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person
