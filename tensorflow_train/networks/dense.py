
from tensorflow_train.layers.layers import concat_channels, conv2d, conv3d
import tensorflow.compat.v1 as tf


class DenseBlock(object):
    def __init__(self, dim, num_layers, num_btnk_filters, num_intermediate_filters, num_output_filters, name, data_format, *args, **kwargs):
        self.dim = dim
        self.num_layers = num_layers
        self.num_btnk_filters = num_btnk_filters
        self.num_intermediate_filters = num_intermediate_filters
        self.num_output_filters = num_output_filters
        self.name = name
        self.data_format = data_format
        self.conv_args = args
        self.conv_kwargs = kwargs

    def conv(self, node, kernel_size, filters, name, is_training):
        raise NotImplementedError

    def transition_layer(self, node, is_training):
        node = self.conv(node, [1] * self.dim, self.num_output_filters, 'transition', is_training)
        return node

    def layer(self, node, is_training):
        node = self.conv(node, [1] * self.dim, self.num_btnk_filters, 'btnk', is_training)
        node = self.conv(node, [3] * self.dim, self.num_intermediate_filters, 'conv', is_training)
        return node

    def dense_block(self, node, is_training):
        with tf.variable_scope(self.name):
            previous_outputs = []
            for current_l in range(self.num_layers):
                with tf.variable_scope('layer' + str(current_l)):
                    if current_l == 0:
                        current_input = node
                    else:
                        current_input = concat_channels(previous_outputs, name='concat', data_format=self.data_format)
                    current_output = self.layer(current_input, is_training)
                    previous_outputs.append(current_output)
            with tf.variable_scope('transition_layer'):
                current_input = concat_channels(previous_outputs, name='concat', data_format=self.data_format)
                output = self.transition_layer(current_input, is_training)
        return output

    def __call__(self, node, is_training):
        return self.dense_block(node, is_training)


class DenseBlock2d(DenseBlock):
    def __init__(self, *args, **kwargs):
        super(DenseBlock2d, self).__init__(2, *args, **kwargs)

    def conv(self, node, kernel_size, filters, name, is_training):
        return conv2d(node,
                      filters,
                      kernel_size,
                      name=name,
                      is_training=is_training,
                      data_format=self.data_format,
                      *self.conv_args,
                      **self.conv_kwargs)


class DenseBlock3d(DenseBlock):
    def __init__(self, *args, **kwargs):
        super(DenseBlock3d, self).__init__(3, *args, **kwargs)

    def conv(self, node, kernel_size, filters, name, is_training):
        return conv3d(node,
                      filters,
                      kernel_size,
                      name=name,
                      is_training=is_training,
                      data_format=self.data_format,
                      *self.conv_args,
                      **self.conv_kwargs)
