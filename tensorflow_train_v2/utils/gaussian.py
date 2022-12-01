
import tensorflow as tf

from tensorflow_train_v2.utils.data_format import get_tf_data_format, get_channel_index, get_channel_size


def gaussian_kernel1d(sigma, filter_shape):
    """
    Calculate a 1d gaussian kernel.
    """
    sigma = tf.convert_to_tensor(sigma)
    coordinates = tf.cast(tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1), sigma.dtype)
    kernel = tf.exp(-0.5 / (sigma ** 2) * coordinates ** 2)
    kernel = kernel / tf.math.reduce_sum(kernel)
    return kernel


def gaussian(image,
             sigma,
             filter_shape=None,
             padding='symmetric',
             constant_values=0,
             data_format='channels_first',
             name=None):
    """
    Gaussian filtering of a 2d or 3d image tensor.
    :param image: The tf image tensor to filter.
    :param sigma: The sigma per dimension. If only a single sigma is given, use same sigma for all dimensions.
    :param filter_shape: The shape of the filter. If None, use sigma to calculate filter shape.
    :param padding: The padding to use before filtering.
    :param constant_values: If padding is constant, use this value for padding.
    :param data_format: 'channels_first' or 'channels_last'
    :param name: The name of the tf operation. If None, use 'gaussian'.
    """
    with tf.name_scope(name or 'gaussian'):
        image = tf.convert_to_tensor(image, name='image')
        dim = image.shape.ndims - 2
        sigma = tf.convert_to_tensor(sigma, name='sigma')
        if sigma.shape.ndims == 0:
            sigma = tf.stack([sigma] * dim)

        if filter_shape is not None:
            filter_shape = tf.convert_to_tensor(filter_shape, name='filter_shape', dtype=tf.int32)
        else:
            filter_shape = tf.cast(tf.math.ceil(tf.cast(sigma, tf.float32) * 4 + 0.5) * 2 + 1, tf.int32)

        # calculate later needed tensor values (must be done before padding!)
        data_format_tf = get_tf_data_format(image, data_format=data_format)
        channel_size = get_channel_size(image, data_format=data_format, as_tensor=False)
        channel_axis = get_channel_index(image, data_format=data_format)

        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.cast(image, tf.float32)

        # calculate gaussian kernels
        sigma = tf.cast(sigma, image.dtype)
        gaussian_kernels = []
        for i in range(dim):
            current_gaussian_kernel = gaussian_kernel1d(sigma[i], filter_shape[i])
            current_gaussian_kernel = tf.reshape(current_gaussian_kernel, [-1 if j == i else 1 for j in range(dim + 2)])
            gaussian_kernels.append(current_gaussian_kernel)

        # pad image for kernel size
        paddings_half = filter_shape // 2
        if data_format == 'channels_first':
            paddings_half = tf.concat([tf.zeros([2], tf.int32), paddings_half], axis=0)
        else:
            paddings_half = tf.concat([tf.zeros([1], tf.int32), paddings_half, tf.zeros([1], tf.int32)], axis=0)
        paddings = tf.stack([paddings_half, paddings_half], axis=1)
        image = tf.pad(image, paddings, mode=padding, constant_values=constant_values)

        # channelwise convolution
        split_inputs = tf.split(image, channel_size, axis=channel_axis, name='split')
        output_list = []
        for i in range(len(split_inputs)):
            current_output = split_inputs[i]
            for current_gaussian_kernel in gaussian_kernels:
                if dim == 2:
                    current_output = tf.nn.conv2d(current_output, current_gaussian_kernel, (1, 1, 1, 1), data_format=data_format_tf, name='conv' + str(i), padding='VALID')
                else:
                    current_output = tf.nn.conv3d(current_output, current_gaussian_kernel, (1, 1, 1, 1, 1), data_format=data_format_tf, name='conv' + str(i), padding='VALID')
            output_list.append(current_output)
        output = tf.concat(output_list, axis=channel_axis, name='concat')

        return tf.cast(output, orig_dtype)
