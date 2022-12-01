
import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import concat_channels, conv2d, max_pool2d, upsample2d, conv3d, max_pool3d, upsample3d

class DownsamplingNetBase(object):
    def __init__(self,
                 num_filters_base,
                 num_levels,
                 downsample_after_last_level=False,
                 double_filters_per_level=False,
                 normalization=None,
                 activation=tf.nn.relu,
                 data_format='channels_first'):
        self.num_filters_base = num_filters_base
        self.num_levels = num_levels
        self.downsample_after_last_level = downsample_after_last_level
        self.double_filters_per_level = double_filters_per_level
        self.normalization = normalization
        self.activation = activation
        self.data_format = data_format

    def num_filters(self, current_level):
        if self.double_filters_per_level:
            return self.num_filters_base * (2 ** current_level)
        else:
            return self.num_filters_base

    def downsample(self, node, current_level, is_training):
        raise NotImplementedError

    def conv(self, node, current_level, postfix, is_training):
        raise NotImplementedError

    def contracting_block(self, node, current_level, is_training):
        raise NotImplementedError

    def contracting(self, node, is_training):
        with tf.variable_scope('downsampling'):
            print('downsampling path')
            for current_level in range(self.num_levels):
                node = self.contracting_block(node, current_level, is_training)
                # do not perform downsampling, if at last level and not self.downsample_after_last_level
                if self.downsample_after_last_level or current_level < self.num_levels - 1:
                    node = self.downsample(node, current_level, is_training)
            return node

    def __call__(self, node, is_training):
        return self.contracting(node, is_training)


class DownsamplingNetBase2D(DownsamplingNetBase):
    def downsample(self, node, current_level, is_training):
        return max_pool2d(node,
                          [2, 2],
                          name='downsample' + str(current_level),
                          data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv2d(node,
                      self.num_filters(current_level),
                      [3, 3], name='conv' + str(current_level) + postfix,
                      activation=self.activation,
                      normalization=self.normalization,
                      is_training=is_training,
                      data_format=self.data_format)
        #node = conv2d(node, self.num_filters(current_level), [3, 3], name='conv' + str(current_level) + '_0' + postfix, activation=self.activation, use_batch_norm=self.use_batch_norm, is_training=self.is_training)
        #return conv2d(node, self.num_filters(current_level), [3, 3], name='conv' + str(current_level) + '_1' + postfix, activation=self.activation, use_batch_norm=self.use_batch_norm, is_training=self.is_training)


class DownsamplingNetBase3D(DownsamplingNetBase):
    def downsample(self, node, current_level, is_training):
        return max_pool3d(node,
                          [2, 2, 2],
                          name='downsample' + str(current_level),
                          data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv3d(node,
                      self.num_filters(current_level),
                      [3, 3, 3],
                      name='conv' + str(current_level) + postfix,
                      activation=self.activation,
                      normalization=self.normalization,
                      is_training=is_training,
                      data_format=self.data_format)


class DownsamplingNet(DownsamplingNetBase):
    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return node


class DownsamplingNet2D(DownsamplingNet, DownsamplingNetBase2D): pass
class DownsamplingNet3D(DownsamplingNet, DownsamplingNetBase3D): pass
