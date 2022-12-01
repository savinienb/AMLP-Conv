
import tensorflow.compat.v1 as tf
from tensorflow_train.layers.initializers import he_initializer, zeros_initializer
from tensorflow_train.utils.print_utils import print_conv_parameters
import numpy as np
from tensorflow_train.utils.data_format import get_tf_data_format_2d, get_tf_data_format_3d
from tensorflow_train.layers.interpolation import upsample2d_linear, upsample3d_linear

from tensorflow_train.layers.layers import debug_print_conv, pad_for_conv
from tensorflow_train.layers.resize_linear import resize_trilinear

def conv2d_bilinear_scale_invariant(inputs,
                                    filters,
                                    kernel_size,
                                    name,
                                    use_autofocus=True,
                                    autofocus_kernel_size=None,
                                    scale_factors=None,
                                    activation=None,
                                    kernel_initializer=he_initializer,
                                    bias_initializer=zeros_initializer,
                                    normalization=None,
                                    is_training=False,
                                    data_format='channels_first',
                                    padding='same',
                                    use_bias=True,
                                    debug_print=debug_print_conv):
    if scale_factors is None:
        scale_factors = [1.0, 0.8, 0.66, 0.57, 0.5]
        #scale_factors = [1.0, 0.66, 0.5]
    data_format_tf = get_tf_data_format_2d(data_format)
    inputs_shape = inputs.get_shape().as_list()
    #if data_format == 'channels_first':
    #    image_size = inputs_shape[2:4]
    #    num_inputs = inputs_shape[1]
    if data_format == 'channels_last':
        image_size = inputs_shape[1:3]
        num_inputs = inputs_shape[3]
    else:
        raise Exception('unsupported data format')

    with tf.variable_scope(name):
        W = tf.get_variable('kernel', kernel_size + [num_inputs, filters], initializer=kernel_initializer, regularizer=tf.nn.l2_loss)
        if use_bias:
            b = tf.get_variable('bias', [filters], initializer=bias_initializer)

        scale_outputs = []
        for scale in scale_factors:
            scaled_size = np.array(image_size) * scale
            curr_outputs = inputs
            if scale != 1.0:
                curr_outputs = tf.image.resize_bilinear(curr_outputs, scaled_size)
            curr_outputs = tf.nn.conv2d(curr_outputs, W, strides=[1, 1, 1, 1], padding=padding.upper(), data_format=data_format_tf, name=name)
            if use_bias:
                curr_outputs = tf.nn.bias_add(curr_outputs, b, data_format=data_format_tf)
            if scale != 1.0:
                curr_outputs = tf.image.resize_bilinear(curr_outputs, image_size)
            scale_outputs.append(curr_outputs)

        scale_outputs_volume = tf.stack(scale_outputs, axis=4, name='stack')

        if use_autofocus:
            with tf.variable_scope('autofocus'):
                #autofocus_conv = tf.layers.conv2d(inputs, kernel_initializer=kernel_initializer, activation=activation, filters=filters, kernel_size=autofocus_kernel_size, data_format=data_format, kernel_regularizer=tf.nn.l2_loss, trainable=is_training, name='conv0', padding=padding)
                autofocus_conv = tf.layers.conv2d(inputs, kernel_initializer=kernel_initializer, filters=len(scale_factors), kernel_size=autofocus_kernel_size, data_format=data_format, kernel_regularizer=tf.nn.l2_loss, trainable=is_training, name='conv1', padding=padding)
                if data_format == 'channels_first':
                    autofocus_conv = tf.nn.softmax(autofocus_conv, axis=1, name='softmax')
                    autofocus_conv = tf.transpose(autofocus_conv, [0, 2, 3, 1])
                    autofocus_conv = tf.expand_dims(autofocus_conv, axis=1)
                else:
                    autofocus_conv = tf.nn.softmax(autofocus_conv, axis=3, name='softmax')
                    autofocus_conv = tf.expand_dims(autofocus_conv, axis=3)
            outputs = tf.reduce_sum(scale_outputs_volume * autofocus_conv, axis=4, keepdims=False, name='reduce_sum')
        else:
            outputs = tf.reduce_max(scale_outputs_volume, axis=4, keepdims=False, name='reduce_max')

        if normalization is not None:
            outputs = normalization(outputs, is_training=is_training, data_format=data_format, name='norm')

        if activation is not None:
            outputs = activation(outputs)

    if debug_print:
        print_conv_parameters(inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              name=name,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              normalization=normalization,
                              is_training=is_training,
                              data_format=data_format,
                              padding=padding,
                              strides=(1, 1),
                              use_bias=use_bias)

    return outputs


def conv2d_bilinear_integer_scale_invariant(inputs,
                                    filters,
                                    kernel_size,
                                    name,
                                    use_autofocus=True,
                                    autofocus_kernel_size=None,
                                    scale_factors=None,
                                    activation=None,
                                    kernel_initializer=he_initializer,
                                    bias_initializer=zeros_initializer,
                                    normalization=None,
                                    is_training=False,
                                    data_format='channels_first',
                                    padding='same',
                                    use_bias=True,
                                    debug_print=debug_print_conv):
    if scale_factors is None:
        scale_factors = [1, 2, 4, 8]
        #scale_factors = [1.0, 0.66, 0.5]
    data_format_tf = get_tf_data_format_2d(data_format)
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        image_size = inputs_shape[2:4]
        channel_axis = 1
    elif data_format == 'channels_last':
        image_size = inputs_shape[1:3]
        channel_axis = 3
    num_inputs = inputs_shape[channel_axis]

    with tf.variable_scope(name):
        W = tf.get_variable('kernel', kernel_size + [num_inputs, filters], initializer=kernel_initializer, regularizer=tf.nn.l2_loss)
        if use_bias:
            b = tf.get_variable('bias', [filters], initializer=bias_initializer)

        scale_outputs = []
        for scale in scale_factors:
            curr_outputs = inputs
            scale_factors_list = [scale] * 2
            if scale != 1:
                curr_outputs = tf.layers.average_pooling2d(curr_outputs, scale_factors_list, scale_factors_list, padding='same', data_format=data_format, name='downsample')
            curr_outputs = tf.nn.conv2d(curr_outputs, W, strides=[1, 1, 1, 1], padding=padding.upper(), data_format=data_format_tf, name=name)
            if use_bias:
                curr_outputs = tf.nn.bias_add(curr_outputs, b, data_format=data_format_tf)
            if scale != 1:
                curr_outputs = upsample2d_linear(curr_outputs, scale_factors_list, name='upsample', data_format=data_format, padding='valid_cropped', output_size_cropped=image_size)
            scale_outputs.append(curr_outputs)

        scale_outputs_volume = tf.stack(scale_outputs, axis=4, name='stack')

        if use_autofocus:
            with tf.variable_scope('autofocus'):
                inputs_avg = tf.reduce_mean(inputs, axis=channel_axis, keepdims=True)
                autofocus_conv = tf.layers.conv2d(inputs_avg, kernel_initializer=kernel_initializer, activation=activation, filters=filters, kernel_size=autofocus_kernel_size, data_format=data_format, kernel_regularizer=tf.nn.l2_loss, trainable=is_training, name='conv0', padding=padding)
                autofocus_conv = tf.layers.conv2d(autofocus_conv, kernel_initializer=kernel_initializer, filters=len(scale_factors), kernel_size=[1, 1], data_format=data_format, kernel_regularizer=tf.nn.l2_loss, trainable=is_training, name='conv1', padding=padding)
                if data_format == 'channels_first':
                    autofocus_conv = tf.nn.softmax(autofocus_conv, axis=1, name='softmax')
                    autofocus_conv = tf.transpose(autofocus_conv, [0, 2, 3, 1])
                    autofocus_conv = tf.expand_dims(autofocus_conv, axis=1)
                else:
                    autofocus_conv = tf.nn.softmax(autofocus_conv, axis=3, name='softmax')
                    autofocus_conv = tf.expand_dims(autofocus_conv, axis=3)
            outputs = tf.reduce_sum(scale_outputs_volume * autofocus_conv, axis=4, keepdims=False, name='reduce_sum')
        else:
            outputs = tf.reduce_max(scale_outputs_volume, axis=4, keepdims=False, name='reduce_max')

        if normalization is not None:
            outputs = normalization(outputs, is_training=is_training, data_format=data_format, name='norm')

        if activation is not None:
            outputs = activation(outputs)

    if debug_print:
        print_conv_parameters(inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              name=name,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              normalization=normalization,
                              is_training=is_training,
                              data_format=data_format,
                              padding=padding,
                              strides=(1, 1),
                              use_bias=use_bias)

    return outputs


def conv3d_bilinear_integer_scale_invariant(inputs,
                                    filters,
                                    kernel_size,
                                    name,
                                    use_autofocus=True,
                                    scale_factors=None,
                                    activation=None,
                                    kernel_initializer=he_initializer,
                                    bias_initializer=zeros_initializer,
                                    normalization=None,
                                    is_training=False,
                                    data_format='channels_first',
                                    padding='same',
                                    use_bias=True,
                                    debug_print=debug_print_conv):
    if scale_factors is None:
        scale_factors = [1, 2, 4, 8]
        #scale_factors = [1.0, 0.66, 0.5]
    data_format_tf = get_tf_data_format_3d(data_format)
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        image_size = inputs_shape[2:5]
        channel_axis = 1
    elif data_format == 'channels_last':
        image_size = inputs_shape[1:4]
        channel_axis = 4
    num_inputs = inputs_shape[channel_axis]

    with tf.variable_scope(name):
        W = tf.get_variable('kernel', kernel_size + [num_inputs, filters], initializer=kernel_initializer, regularizer=tf.nn.l2_loss)
        if use_bias:
            b = tf.get_variable('bias', [filters], initializer=bias_initializer)
            if data_format == 'channels_first':
                b_reshaped = b[None, :, None, None, None]
            else:
                b_reshaped = b[None, None, None, None, :]

        scale_outputs = []
        # for scale in scale_factors:
        #     curr_outputs = inputs
        #     scale_factors_list = [scale] * 3
        #     if scale != 1:
        #         curr_outputs = tf.layers.average_pooling3d(curr_outputs, scale_factors_list, scale_factors_list, padding='same', data_format=data_format, name='downsample')
        #     curr_outputs = tf.nn.conv3d(curr_outputs, W, strides=[1, 1, 1, 1, 1], padding=padding.upper(), data_format=data_format_tf, name=name)
        #     if use_bias:
        #         curr_outputs += b_reshaped  #tf.nn.bias_add(curr_outputs, b, data_format=data_format_tf)
        #     if scale != 1:
        #         curr_outputs = upsample3d_linear(curr_outputs, scale_factors_list, name='upsample', data_format=data_format)#resize_trilinear(curr_outputs, scale_factors_list, name='upsample', data_format=data_format)
        #     scale_outputs.append(curr_outputs)
        #
        # scale_outputs_volume = tf.stack(scale_outputs, axis=5, name='stack')
        #
        # if use_autofocus:
        #     with tf.variable_scope('autofocus'):
        #         autofocus_var = tf.get_variable('autofocus_softmax', initializer=tf.initializers.truncated_normal, shape=[filters, len(scale_factors)])
        #         autofocus_var_softmax = tf.nn.softmax(autofocus_var, axis=1)
        #         if data_format == 'channels_first':
        #             autofocus_var_softmax = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(autofocus_var_softmax, axis=0), axis=2), axis=2), axis=2)
        #         else:
        #             autofocus_var_softmax = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(autofocus_var_softmax, axis=0), axis=0), axis=0), axis=0)
        #     outputs = tf.reduce_sum(scale_outputs_volume * autofocus_var_softmax, axis=5, keepdims=False, name='reduce_sum')
        # else:
        #     outputs = tf.reduce_max(scale_outputs_volume, axis=5, keepdims=False, name='reduce_max')
        if use_autofocus:
            with tf.variable_scope('autofocus'):
                autofocus_var = tf.get_variable('autofocus_softmax', initializer=tf.initializers.truncated_normal, shape=[filters, len(scale_factors)])
                autofocus_var_softmax = tf.nn.softmax(autofocus_var, axis=1)

        outputs = None
        for i, scale in enumerate(scale_factors):
            curr_outputs = inputs
            scale_factors_list = [scale] * 3
            if scale != 1:
                curr_outputs = tf.layers.average_pooling3d(curr_outputs, scale_factors_list, scale_factors_list, padding='same', data_format=data_format, name='downsample')
            curr_outputs, padding_for_conv = pad_for_conv(curr_outputs, kernel_size, 'pad', padding, data_format)
            curr_outputs = tf.nn.conv3d(curr_outputs, W, strides=[1, 1, 1, 1, 1], padding=padding_for_conv.upper(), data_format=data_format_tf, name=name)
            if use_bias:
                curr_outputs += b_reshaped  #tf.nn.bias_add(curr_outputs, b, data_format=data_format_tf)
            if use_autofocus:
                curr_autofocus = autofocus_var_softmax[:, i]
                if data_format == 'channels_first':
                    curr_autofocus = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(curr_autofocus, axis=0), axis=2), axis=2), axis=2)
                else:
                    curr_autofocus = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(curr_autofocus, axis=0), axis=0), axis=0), axis=0)
                curr_outputs *= curr_autofocus
            if scale == 1:
                outputs = curr_outputs
            if scale != 1:
                curr_outputs = upsample3d_linear(curr_outputs, scale_factors_list, name='upsample', data_format=data_format, padding='valid_cropped')
                #curr_outputs = resize_trilinear(curr_outputs, scale_factors_list, name='upsample', data_format=data_format)
                outputs += curr_outputs
            #scale_outputs.append(curr_outputs)

        # scale_outputs_volume = tf.stack(scale_outputs, axis=5, name='stack')
        #
        # if use_autofocus:
        #     with tf.variable_scope('autofocus'):
        #         autofocus_var = tf.get_variable('autofocus_softmax', initializer=tf.initializers.truncated_normal, shape=[filters, len(scale_factors)])
        #         autofocus_var_softmax = tf.nn.softmax(autofocus_var, axis=1)
        #         if data_format == 'channels_first':
        #             autofocus_var_softmax = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(autofocus_var_softmax, axis=0), axis=2), axis=2), axis=2)
        #         else:
        #             autofocus_var_softmax = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(autofocus_var_softmax, axis=0), axis=0), axis=0), axis=0)
        #     outputs = tf.reduce_sum(scale_outputs_volume * autofocus_var_softmax, axis=5, keepdims=False, name='reduce_sum')
        # else:
        #     outputs = tf.reduce_max(scale_outputs_volume, axis=5, keepdims=False, name='reduce_max')


        if normalization is not None:
            outputs = normalization(outputs, is_training=is_training, data_format=data_format, name='norm')

        if activation is not None:
            outputs = activation(outputs)

    if debug_print:
        print_conv_parameters(inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              name=name,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              normalization=normalization,
                              is_training=is_training,
                              data_format=data_format,
                              padding=padding,
                              strides=(1, 1, 1),
                              use_bias=use_bias)

    return outputs


def conv2d_dilation_scale_invariant(inputs,
                                    filters,
                                    kernel_size,
                                    name,
                                    use_autofocus=True,
                                    autofocus_kernel_size=None,
                                    dilation_rates=None,
                                    activation=None,
                                    kernel_initializer=he_initializer,
                                    bias_initializer=zeros_initializer,
                                    normalization=None,
                                    is_training=False,
                                    data_format='channels_first',
                                    padding='same',
                                    use_bias=True,
                                    debug_print=debug_print_conv):

    dilation_rates = dilation_rates or [1, 2, 3, 4]
    autofocus_kernel_size = autofocus_kernel_size or kernel_size

    data_format_tf = get_tf_data_format_2d(data_format)
    inputs_shape = inputs.get_shape().as_list()

    num_inputs = inputs_shape[1] if data_format == 'channels_first' else inputs_shape[3]

    with tf.variable_scope(name):
        W = tf.get_variable('kernel', kernel_size + [num_inputs, filters], initializer=kernel_initializer, regularizer=tf.nn.l2_loss)
        if use_bias:
            b = tf.get_variable('bias', [filters], initializer=bias_initializer)

        scale_outputs = []
        for dilation_rate in dilation_rates:
            curr_outputs = inputs
            dilations = [1, 1, dilation_rate, dilation_rate] if data_format == 'channels_first' else [1, dilation_rate, dilation_rate, 1]
            curr_outputs = tf.nn.conv2d(curr_outputs, W, strides=[1, 1, 1, 1], dilations=dilations, padding=padding.upper(), data_format=data_format_tf)
            if use_bias:
                curr_outputs = tf.nn.bias_add(curr_outputs, b, data_format=data_format_tf)
            scale_outputs.append(curr_outputs)

        scale_outputs_volume = tf.stack(scale_outputs, axis=4, name='stack')

        if use_autofocus:
            with tf.variable_scope('autofocus'):
                #autofocus_conv = tf.layers.conv2d(inputs, kernel_initializer=kernel_initializer, activation=activation, filters=filters, kernel_size=autofocus_kernel_size, data_format=data_format, kernel_regularizer=tf.nn.l2_loss, trainable=is_training, name='conv0', padding=padding)
                autofocus_conv = tf.layers.conv2d(inputs, kernel_initializer=kernel_initializer, filters=len(dilation_rates), kernel_size=autofocus_kernel_size, data_format=data_format, kernel_regularizer=tf.nn.l2_loss, trainable=is_training, name='conv1', padding=padding)
                if data_format == 'channels_first':
                    autofocus_conv = tf.nn.softmax(autofocus_conv, axis=1, name='softmax')
                    autofocus_conv = tf.transpose(autofocus_conv, [0, 2, 3, 1])
                    autofocus_conv = tf.expand_dims(autofocus_conv, axis=1)
                else:
                    autofocus_conv = tf.nn.softmax(autofocus_conv, axis=3, name='softmax')
                    autofocus_conv = tf.expand_dims(autofocus_conv, axis=3)
            outputs = tf.reduce_sum(scale_outputs_volume * autofocus_conv, axis=4, keepdims=False, name='reduce_sum')
        else:
            outputs = tf.reduce_max(scale_outputs_volume, axis=4, keepdims=False, name='reduce_max')

        if normalization is not None:
            outputs = normalization(outputs, is_training=is_training, data_format=data_format, name='norm')

        if activation is not None:
            outputs = activation(outputs)

    if debug_print:
        print_conv_parameters(inputs=inputs,
                              outputs=outputs,
                              kernel_size=kernel_size,
                              name=name,
                              activation=activation,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              normalization=normalization,
                              is_training=is_training,
                              data_format=data_format,
                              padding=padding,
                              strides=(1, 1),
                              use_bias=use_bias)

    return outputs


def conv2d_rotation_invariant(inputs,
                              filters,
                              kernel_size,
                              name,
                              num_rotations=8,
                              activation=None,
                              kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal'),
                              bias_initializer=tf.zeros_initializer(),
                              use_batch_norm=False,
                              is_training=False,
                              data_format='channels_first'):

    data_format_tf = get_tf_data_format_2d(data_format)
    inputs_shape = inputs.get_shape().as_list()
    #if data_format == 'channels_first':
    #    image_size = inputs_shape[2:4]
    #    num_inputs = inputs_shape[1]
    if data_format == 'channels_last':
        image_size = inputs_shape[1:3]
        num_inputs = inputs_shape[3]
    else:
        raise Exception('unsupported data format')

    W = tf.get_variable(name + '/kernel', [kernel_size[0], kernel_size[1], num_inputs, filters], initializer=kernel_initializer, regularizer=tf.nn.l2_loss)
    b = tf.get_variable(name + '/bias', [filters], initializer=bias_initializer)

    scale_outputs = []
    for scale in scale_factors:
        curr_size = np.array(image_size) * scale
        curr_outputs = inputs
        if scale != 1.0:
            curr_outputs = tf.image.resize_bilinear(curr_outputs, curr_size)
        curr_outputs = tf.nn.conv2d(curr_outputs, W, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format_tf, name=name)
        curr_outputs = tf.nn.bias_add(curr_outputs, b, data_format=data_format_tf)
        if scale != 1.0:
            curr_outputs = tf.image.resize_bilinear(curr_outputs, image_size)
        scale_outputs.append(curr_outputs)

    scale_outputs_volume = tf.stack(scale_outputs, axis=4, name=name+'/stack')
    x = tf.reduce_max(scale_outputs_volume, axis=4, keepdims=False, name=name+'/reduce_max')

    if use_batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, data_format=data_format_tf, fused=True, scope=name+'/bn')

    if activation is not None:
        return activation(x)

    return x
