
import tensorflow as tf
from keras.layers import Conv3D
from tensorflow.python.keras.regularizers import l2

from tensorflow_train_v2.layers.upsample import upsample_linear, upsample_cubic, upsample_lanczos
from tensorflow_train_v2.utils.data_format import get_batch_channel_image_size_from_shape_tuple, create_tensor_shape_tuple


class Sequential(tf.keras.layers.Layer):
    """
    A keras layer that applies a list of encapsulated layers sequentially.
    """
    def __init__(self, layers, *args, **kwargs):
        """
        Initializer.
        :param layers: The list of layers to apply sequentially.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(Sequential, self).__init__(*args, **kwargs)
        self.layers = layers

    def build(self, input_shape):
        """
        Build the layer.
        :param input_shape: The input shape.
        """
        current_shape = input_shape
        for layer in self.layers:
            layer.build(current_shape)
            current_shape = tf.TensorShape(layer.compute_output_shape(tuple(current_shape.as_list())))

    def call(self, inputs, **kwargs):
        """
        Call the internal layers sequentially for the given inputs.
        :param inputs: The layer inputs.
        :param kwargs: **kwargs
        :return: The output of the last layer.
        """
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class ConcatChannels(tf.keras.layers.Layer):
    """
    Concat the channel dimension.
    """
    def __init__(self, data_format=None, *args, **kwargs):
        """
        Initializer
        :param data_format: The data_format.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(ConcatChannels, self).__init__(*args, **kwargs)
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        """
        Computes the output_shape.
        :param input_shape: The input shape.
        :return: The output shape.
        """
        if input_shape is None:
            return None
        all_channel_sizes = [get_batch_channel_image_size_from_shape_tuple(s, self.data_format)[1] if s is not None else None for s in input_shape]
        if None in all_channel_sizes:
            return None
        return sum(all_channel_sizes)

    def call(self, inputs, **kwargs):
        """
        Concatenate the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The concatenated outputs.
        """
        axis = 1 if self.data_format == 'channels_first' else -1
        return tf.concat(inputs, axis=axis)


class UpSamplingBase(tf.keras.layers.Layer):
    """
    UpSampling base layer.
    """
    def __init__(self, dim, size, data_format, *args, **kwargs):
        """
        Initializer.
        :param size: The scaling factors. Must be integer.
        :param data_format: The data_format.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(UpSamplingBase, self).__init__(*args, **kwargs)
        self.dim = dim
        self.size = size
        if self.dim != len(self.size):
            raise ValueError(f'dim and size parameter do not agree, dim: {dim}, size: {size}')
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        """
        Computes the output_shape.
        :param input_shape: The input shape.
        :return: The output shape.
        """
        if input_shape is None:
            return None
        if not isinstance(input_shape, tuple):
            # when input_shape is a list, multiple inputs were defined
            raise ValueError('UpSampling layers only allow a single input')
        if len(input_shape) != self.dim + 2:
            raise ValueError(f'Dimension of input tensor is invalid, len(input_shape): {len(input_shape)}, expected: {self.dim + 2}')
        batch_size, channel_size, image_size = get_batch_channel_image_size_from_shape_tuple(input_shape, self.data_format)
        new_image_size = tuple([size * scale if size is not None else None for size, scale in zip(image_size, self.size)])
        return create_tensor_shape_tuple(batch_size, channel_size, new_image_size, self.data_format)


class UpSampling2DLinear(UpSamplingBase):
    """
    UpSampling 3D with linear interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling2DLinear, self).__init__(2, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_linear(inputs, self.size, self.data_format)


class UpSampling3DLinear(UpSamplingBase):
    """
    UpSampling 3D with linear interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling3DLinear, self).__init__(3, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_linear(inputs, self.size, self.data_format)


class UpSampling2DCubic(UpSamplingBase):
    """
    UpSampling 2D with cubic interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling2DCubic, self).__init__(2, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_cubic(inputs, self.size, self.data_format)


class UpSampling3DCubic(UpSamplingBase):
    """
    UpSampling 3D with cubic interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling3DCubic, self).__init__(3, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_cubic(inputs, self.size, self.data_format)


class UpSampling2DLanczos(UpSamplingBase):
    """
    UpSampling 2D with lanczos interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling2DLanczos, self).__init__(2, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_lanczos(inputs, self.size, self.data_format)


class UpSampling3DLanczos(UpSamplingBase):
    """
    UpSampling 3D with lanczos interpolation.
    """
    def __init__(self, size, data_format, *args, **kwargs):
        """
        Initializer.
        """
        super(UpSampling3DLanczos, self).__init__(3, size, data_format, *args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        Upsample the given inputs.
        :param inputs: The inputs.
        :param kwargs: *kwargs
        :return: The upsampled outputs.
        """
        return upsample_lanczos(inputs, self.size, self.data_format)


class Mixer(tf.keras.layers.Layer):
    """
    UpSampling base layer.
    """

    def __init__(self, num_filters_base, kernel_size, name, activation, data_format, kernel_initializer, kernel_regularizer, padding, reduction=8, *args, **kwargs):
        """
        Initializer.
        :param size: The scaling factors. Must be integer.
        :param data_format: The data_format.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(Mixer, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.data_format = data_format
        self.kernel_size = 5
        self.nameio = name
        self.activation = activation
        self.data_format = data_format
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.padding = padding
        self.reduce_factor = 8
        self.reduce_factor2 = 1

    def directional(self, x, axis, reverse=False):
        shape = [1, 1, 1, 1, 1]
        shape[axis] = x.shape[axis]
        # temp = tf.ones(shape)
        sum_val = tf.cumsum(x, axis, reverse) - x
        # denom = tf.cumsum(temp,axis,reverse)-1+1e-5
        return sum_val  # /denom

    def instance_norm(self, x, axis=[-1, -2, -3]):
        mean, variance = tf.nn.moments(x, axis, keepdims=True)
        out = (x - mean) / tf.sqrt(variance + 1e-5)
        return out

    def build(self, training):
        self.conv = Conv3D(self.num_filters_base // self.reduce_factor2, self.kernel_size, name=self.nameio + '_conv', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        
        if 'expand' in  self.nameio:
            if '3_conv_0expand' in self.nameio:
                self.num_filters_base *= 2
            elif '2_conv_0expand' in self.nameio:
                self.num_filters_base *= 2
            elif '1_conv_0expand' in self.nameio:
                self.num_filters_base *= 2
            elif '0_conv_0expand' in self.nameio:
                self.num_filters_base *= 2
        self.up = Conv3D(self.num_filters_base // self.reduce_factor, 1, name=self.nameio + '_up', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.down = Conv3D(self.num_filters_base // self.reduce_factor, 1, name=self.nameio + '_down', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.left = Conv3D(self.num_filters_base // self.reduce_factor, 1, name=self.nameio + '_left', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding) 
        
        self.up1 = Conv3D(self.num_filters_base , 1, name=self.nameio + '_up1', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.down1 = Conv3D(self.num_filters_base, 1, name=self.nameio + '_down1', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.left1 = Conv3D(self.num_filters_base, 1, name=self.nameio + '_left1', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding) 
        

    def call(self, x, **kwargs):
        """
        Computes the output_shape.
        :param input_shape: The input shape.
        :return: The output shape.
        """
        
        up = self.up(x)
        down = self.down(x)
        left = self.left(x)
        
        up = tf.nn.relu(up)
        down = tf.nn.relu(down)
        left = tf.nn.relu(left)
        
        left = tf.reduce_sum(left,axis=-1, keepdims = True)
        up = tf.reduce_sum(up,axis=-2,keepdims = True)
        down = tf.reduce_sum(down, axis=-3,keepdims = True)
        
        up = self.up1(up)
        down = self.down1(down)
        left = self.left1(left)


        att = up + down + left 
        att = tf.nn.sigmoid(att)
        x *= att
        
        x = self.conv(x)
        x = self.instance_norm(x) + x

        return x



