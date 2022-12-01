
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, AveragePooling3D, Add, AlphaDropout, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow_train_v2.networks.unet_base import UnetBase
from tensorflow_train_v2.layers.layers import Sequential, ConcatChannels, UpSampling3DLinear, UpSampling3DCubic, Amlp_Conv
from tensorflow_train_v2.layers.initializers import he_initializer, selu_initializer

from tensorflow.keras import initializers

class UnetAvgLinear3D(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    """
    def __init__(self,
                 num_filters_base,
                 repeats=2,
                 dropout_ratio=0.0,
                 kernel_size=None,
                 activation=tf.nn.relu,
                 kernel_initializer=he_initializer,
                 alpha_dropout=False,
                 data_format='channels_first',
                 padding='same',
                 *args, **kwargs):
        super(UnetAvgLinear3D, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 3
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.alpha_dropout = alpha_dropout
        self.data_format = data_format
        self.padding = padding
        self.init_layers()

    def downsample(self, current_level):
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return AveragePooling3D([2] * 3, data_format=self.data_format)

    def upsample(self, current_level):
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return UpSampling3DLinear([2] * 3, data_format=self.data_format)

    def combine(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return ConcatChannels(data_format=self.data_format)
        #return Add()

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.dropout_ratio > 0:
                if self.alpha_dropout:
                    layers.append(AlphaDropout(self.dropout_ratio))
                else:
                    layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='contracting' + str(current_level))

    def expanding_block(self, current_level):
        """
        Create and return the expanding block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.conv(current_level, str(i)))
            if self.dropout_ratio > 0:
                if self.alpha_dropout:
                    layers.append(AlphaDropout(self.dropout_ratio))
                else:
                    layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='expanding' + str(current_level))

    def conv(self, current_level, postfix):
        """
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        return Conv3D(self.num_filters_base,
                      self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=l2(l=1.0),
                      padding=self.padding)



class Unet(tf.keras.Model):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self, num_labels, num_filters_base=64, num_levels=4, activation='relu', data_format='channels_first', padding='same', dropout_ratio=0.0, **kwargs):
        """
        The spatial configuration net.
        :param input: Input tensor.
        :param num_labels: Number of outputs.
        :param is_training: True, if training network.
        :param data_format: 'channels_first' or 'channels_last'
        :param actual_network: The actual u-net instance used as the local appearance network.
        :param padding: Padding parameter passed to the convolution operations.
        :param spatial_downsample: Downsamping factor for the spatial configuration stage.
        :param args: Not used.
        :param kwargs: Not used.
        :return: heatmaps, local_heatmaps, spatial_heatmaps
        """
        super(Unet, self).__init__()
        self.data_format = data_format
        num_filters_base = num_filters_base
        if activation == 'relu':
            activation_fn = tf.nn.relu
        elif activation == 'lrelu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        self.unet = UnetAvgLinear3D(num_filters_base=num_filters_base, num_levels=num_levels, kernel_initializer=he_initializer, activation=activation_fn, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.prediction = Sequential([Conv3D(num_labels, [1] * 3, name='prediction', kernel_initializer=he_initializer, activation=None, data_format=data_format, padding=padding),
                                    Activation(None, dtype='float32', name='prediction')])
        self.single_output = num_labels == 1


    def call(self, inputs, training, **kwargs):
        node = self.unet(inputs, training=training)
        prediction = self.prediction(node, training=training)
        if self.single_output:
            return prediction
        else:
            return prediction, prediction, prediction, prediction, prediction


class UnetAMLP(tf.keras.Model):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self, num_labels, num_filters_base=32, num_levels=4, activation='relu', data_format='channels_first', padding='same', dropout_ratio=0.0, **kwargs):
        """
        The spatial configuration net.
        :param input: Input tensor.
        :param num_labels: Number of outputs.
        :param is_training: True, if training network.
        :param data_format: 'channels_first' or 'channels_last'
        :param actual_network: The actual u-net instance used as the local appearance network.
        :param padding: Padding parameter passed to the convolution operations.
        :param spatial_downsample: Downsamping factor for the spatial configuration stage.
        :param args: Not used.
        :param kwargs: Not used.
        :return: heatmaps, local_heatmaps, spatial_heatmaps
        """
        super(UnetAMLP, self).__init__()
        self.data_format = data_format
        num_filters_base = num_filters_base
        if activation == 'relu':
            activation_fn = tf.nn.relu
        elif activation == 'lrelu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        self.unet = UnetAMLPAvgLinear3D(num_filters_base=num_filters_base, num_levels=num_levels, kernel_initializer=initializers.RandomNormal(stddev=0.01), activation=activation_fn, dropout_ratio=dropout_ratio, data_format=data_format, padding=padding)
        self.prediction = Sequential([Conv3D(num_labels, [1] * 3, name='prediction', kernel_initializer=he_initializer, activation=None, data_format=data_format, padding=padding),
                                    Activation(None, dtype='float32', name='prediction')])
        self.single_output = num_labels == 1


    def call(self, inputs, training, **kwargs):
        node = self.unet(inputs, training=training)
        prediction = self.prediction(node, training=training)
        if self.single_output:
            return prediction
        else:
            return prediction, prediction, prediction, prediction, prediction


class AMLPLine(tf.keras.Model):
    """
    The SpatialConfigurationNet.
    """
    def __init__(self, num_labels, num_filters_base = 54, activation='relu', data_format='channels_first', padding='same', dropout_ratio=0.0, **kwargs):
        """
        The spatial configuration net.
        :param input: Input tensor.
        :param num_labels: Number of outputs.
        :param is_training: True, if training network.
        :param data_format: 'channels_first' or 'channels_last'
        :param actual_network: The actual u-net instance used as the local appearance network.
        :param padding: Padding parameter passed to the convolution operations.
        :param spatial_downsample: Downsamping factor for the spatial configuration stage.
        :param args: Not used.
        :param kwargs: Not used.
        :return: heatmaps, local_heatmaps, spatial_heatmaps
        """
        super(AMLPLine, self).__init__()
        self.data_format = data_format
        self.kernel_initializer = kernel_initializer

        self.num_filters_base = num_filters_base
        self.kernel_size = [5,5,5]

        if activation == 'relu':
            activation_fn = tf.nn.relu
        elif activation == 'lrelu':
            activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.1)

        self.l0 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_0', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l1 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_1', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l2 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_2', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l3 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_3', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l4 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_4', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l5 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_5', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l6 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_6', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l7 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_7', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l8 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_8', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l9 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_9', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.l10 = Amlp_Conv(self.num_filters_base, self.kernel_size, name='amlp_conv_10', activation=self.activation, data_format=self.data_format, kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(l=1.0), padding=self.padding)
        self.prediction = Sequential([Conv3D(num_labels, [1] * 3, name='prediction', kernel_initializer=he_initializer, activation=None, data_format=data_format, padding=padding),
                                    Activation(None, dtype='float32', name='prediction')])
        self.single_output = num_labels == 1

    def call(self, inputs, training, **kwargs):
        node = self.l0(inputs, training=training)
        node = self.l1(node, training=training)
        node = self.l2(node, training=training)
        node = self.l3(node, training=training)
        node = self.l4(node, training=training)
        node = self.l5(node, training=training)
        node = self.l6(node, training=training)
        node = self.l7(node, training=training)
        node = self.l8(node, training=training)
        node = self.l9(node, training=training)
        node = self.l10(node, training=training)
        prediction = self.prediction(node, training=training)
        if self.single_output:
            return prediction
        else:
            return prediction, prediction, prediction, prediction, prediction


class UnetAMLPAvgLinear3D(UnetBase):
    """
    U-Net with average pooling and linear upsampling.
    """
    def __init__(self,
                 num_filters_base,
                 repeats=2,
                 dropout_ratio=0.0,
                 kernel_size=None,
                 activation=tf.nn.relu,
                 kernel_initializer=he_initializer,
                 alpha_dropout=False,
                 data_format='channels_first',
                 padding='same',
                 *args, **kwargs):
        super(UnetAMLPAvgLinear3D, self).__init__(*args, **kwargs)
        self.num_filters_base = num_filters_base
        self.repeats = repeats
        self.dropout_ratio = dropout_ratio
        self.kernel_size = kernel_size or [3] * 3
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.alpha_dropout = alpha_dropout
        self.data_format = data_format
        self.padding = padding
        self.postfix = 0
        self.init_layers()

    def downsample(self, current_level):
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return AveragePooling3D([2] * 3, data_format=self.data_format)

    def upsample(self, current_level):
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return UpSampling3DLinear([2] * 3, data_format=self.data_format)

    def combine(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return ConcatChannels(data_format=self.data_format)
        #return Add()

    def add(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return AddChannels(data_format=self.data_format)
        #return Add()

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.amlp_conv(current_level, str(i)))
            if self.dropout_ratio > 0:
                if self.alpha_dropout:
                    layers.append(AlphaDropout(self.dropout_ratio))
                else:
                    layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='contracting' + str(current_level))

    def expanding_block(self, current_level):
        """
        Create and return the expanding block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        layers = []
        for i in range(self.repeats):
            layers.append(self.amlp_conv(current_level, str(i)))
            if self.dropout_ratio > 0:
                if self.alpha_dropout:
                    layers.append(AlphaDropout(self.dropout_ratio))
                else:
                    layers.append(Dropout(self.dropout_ratio))
        return Sequential(layers, name='expanding' + str(current_level))

    def layer_norm(input,axis=-1,shape=False):
      mean, variance = tf.nn.moments(input, [axis], keep_dims=True)
      normalized_y = (input-mean) / tf.sqrt(variance + 1e-5)
      out = normalized_y
      if shape is not False:
        out = tf.reshape(out,shape)
      return out

    def conv(self, current_level, postfix):
        """
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        return Conv3D(self.num_filters_base,
                      self.kernel_size,
                      name='conv' + postfix,
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=l2(l=1.0),
                      padding=self.padding)

    def amlp_conv(self, current_level, postfix):
        """
        Create and return a convolution layer for the current level with the current postfix.
        :param current_level: The current level.
        :param postfix:
        :return:
        """
        self.postfix += 1
        return Amlp_Conv(self.num_filters_base,
                      self.kernel_size,
                      name='conv' + str(self.postfix-1),
                      activation=self.activation,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      kernel_regularizer=l2(l=1.0),
                      padding=self.padding)

