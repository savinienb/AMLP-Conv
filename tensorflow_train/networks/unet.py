
import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import conv2d, conv3d, concat_channels, add, dropout, max_pool2d, max_pool3d, avg_pool2d, avg_pool3d, upsample2d, upsample3d
from tensorflow_train.layers.resize_linear import resize_bilinear, resize_trilinear
from tensorflow_train.networks.unet_base import UnetBase, UnetBase2D, UnetBase3D
from tensorflow_train.utils.data_format import get_image_dimension


class UnetGeneral(UnetBase):
    def __init__(self, conv_repeat=2, parallel_conv_repeat=0, use_addition=False, use_avg_pooling=False, use_linear_upsampling=False, *args, **kwargs):
        super(UnetGeneral, self).__init__(*args, **kwargs)
        self.conv_repeat = conv_repeat
        self.parallel_conv_repeat = parallel_conv_repeat
        self.use_addition = use_addition
        self.use_avg_pooling = use_avg_pooling
        self.use_linear_upsampling = use_linear_upsampling

    def downsample(self, node, current_level, is_training):
        dim = get_image_dimension(node)
        if dim == 2:
            if self.use_avg_pooling:
                return avg_pool2d(node, [2, 2], name='downsample_avg_pool' + str(current_level), data_format=self.data_format)
            else:
                return max_pool2d(node, [2, 2], name='downsample_max_pool' + str(current_level), data_format=self.data_format)
        else:
            if self.use_avg_pooling:
                return avg_pool3d(node, [2, 2, 2], name='downsample_avg_pool' + str(current_level), data_format=self.data_format)
            else:
                return max_pool3d(node, [2, 2, 2], name='downsample_max_pool' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        dim = get_image_dimension(node)
        if dim == 2:
            if self.use_linear_upsampling:
                return resize_bilinear(node, [2, 2], name='upsample_linear' + str(current_level), data_format=self.data_format)
            else:
                return upsample2d(node, [2, 2], name='upsample_nearest' + str(current_level), data_format=self.data_format)
        else:
            if self.use_linear_upsampling:
                return resize_trilinear(node, [2, 2, 2], name='upsample_linear' + str(current_level), data_format=self.data_format)
            else:
                return upsample3d(node, [2, 2, 2], name='upsample_nearest' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        dim = get_image_dimension(node)
        if dim == 2:
            return conv2d(node,
                          self.num_filters(current_level),
                          [3, 3],
                          name='conv' + postfix,
                          activation=self.activation,
                          kernel_initializer=self.kernel_initializer,
                          normalization=self.normalization,
                          is_training=is_training,
                          data_format=self.data_format,
                          padding=self.padding)
        else:
            return conv3d(node,
                          self.num_filters(current_level),
                          [3, 3, 3],
                          name='conv' + postfix,
                          activation=self.activation,
                          kernel_initializer=self.kernel_initializer,
                          normalization=self.normalization,
                          is_training=is_training,
                          data_format=self.data_format,
                          padding=self.padding)

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        if self.use_addition:
            return add([parallel_node, upsample_node], name='add' + str(current_level))
        else:
            return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        for i in range(self.conv_repeat):
            node = self.conv(node, current_level, str(i), is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        for i in range(self.parallel_conv_repeat):
            node = self.conv(node, current_level, str(i), is_training)
        return node

    def expanding_block(self, node, current_level, is_training):
        for i in range(self.conv_repeat):
            node = self.conv(node, current_level, str(i), is_training)
        return node


class UnetClassic(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node


class UnetAdd(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node


class UnetParallel(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

    def contracting_block(self, node, current_level, is_training):
        return self.conv(node, current_level, '', is_training)

    def parallel_block(self, node, current_level, is_training):
        return self.conv(node, current_level, '', is_training)

    def expanding_block(self, node, current_level, is_training):
        return self.conv(node, current_level, '', is_training)


class UnetParallelDropout(UnetBase):
    def __init__(self,
                 num_filters_base,
                 num_levels,
                 dropout_rate,
                 double_filters_per_level=False,
                 normalization=None,
                 activation=tf.nn.relu):
        super(UnetParallelDropout, self).__init__(num_filters_base,
                                                  num_levels,
                                                  double_filters_per_level,
                                                  normalization,
                                                  activation)
        self.dropout_rate = dropout_rate

    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return add([parallel_node, upsample_node], name='add' + str(current_level))

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return dropout(node, self.dropout_rate, name='dropout', is_training=is_training)

    def parallel_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return dropout(node, self.dropout_rate, name='dropout', is_training=is_training)

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '', is_training)
        return dropout(node, self.dropout_rate, name='dropout', is_training=is_training)


class UnetClassic2D(UnetClassic, UnetBase2D): pass
class UnetClassic3D(UnetClassic, UnetBase3D): pass
class UnetAdd2D(UnetAdd, UnetBase2D): pass
class UnetAdd3D(UnetAdd, UnetBase3D): pass
class UnetParallel2D(UnetParallel, UnetBase2D): pass
class UnetParallel3D(UnetParallel, UnetBase3D): pass
class UnetParallelDropout2D(UnetParallelDropout, UnetBase2D): pass
class UnetParallelDropout3D(UnetParallelDropout, UnetBase3D): pass