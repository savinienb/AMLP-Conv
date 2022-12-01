import tensorflow as tf
from tensorflow_train_v2.layers.layers import UpSampling3DLinear, UpSampling3DCubic
from tensorflow.keras.layers import AveragePooling3D
import utils

def salt_pepper_3D(image, rate, scales=[1], data_format='channels_first', seed=None):
    image = tf.convert_to_tensor(image, name="image")
    rate = tf.convert_to_tensor(rate, dtype=image.dtype, name="rate")



    # random_tensor = tf.random.uniform(x.shape, seed=seed, dtype=x.dtype)
    # utils.io.image.write_np(random_tensor, 'debug_wo_resampling.nii.gz')

    # image_shape = tf.shape(image)
    # channel_axis = 1 if data_format == 'channels_first' else -1

    # scales = [1, 2, 4, 8]
    ret = image
    for scale in scales:
        # cur_shape = []
        # for idx, value in enumerate(image_shape):
        #     if idx != 0 and idx != channel_axis:
        #         assert value % scale == 0, f'Image size {value} is not divisible by scale {scale}'
        #         cur_shape.append(value // scale)
        #     else:
        #         cur_shape.append(value)

        # cur_shape = tf.convert_to_tensor(cur_shape)

        # dims_to_scale = tf.convert_to_tensor([True] * len(image_shape))
        # dims_to_scale[0] = False
        # dims_to_scale[channel_axis] = False

        pooling_layer = AveragePooling3D([scale] * 3, data_format=data_format)
        upsampling_layer = UpSampling3DLinear([scale] * 3, data_format=data_format)


        dummy = tf.zeros_like(image)
        cur_shape = tf.shape(pooling_layer(dummy))

        random_tensor = tf.random.uniform(cur_shape, seed=seed, dtype=image.dtype)



        # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
        # float to be selected, hence we use a >= comparison.
        zeros_mask = tf.cast(random_tensor < rate / 2, image.dtype)
        ones_mask = tf.cast(random_tensor >= 1 - rate / 2, image.dtype)

        # utils.io.image.write_np(zeros_mask, 'debug/debug_zeros_mask_small.nii.gz')
        # utils.io.image.write_np(ones_mask, 'debug/debug_ones_mask_small.nii.gz')
        zeros_mask = upsampling_layer(zeros_mask)
        ones_mask = upsampling_layer(ones_mask)

        # utils.io.image.write_np(zeros_mask, 'debug/debug_zeros_mask.nii.gz')
        # utils.io.image.write_np(ones_mask, 'debug/debug_ones_mask.nii.gz')
        ret = (1 - (1 - ret) * (1 - ones_mask)) * (1 - zeros_mask)
    # utils.io.image.write_np(tf.cast(image, tf.float32), 'debug/image.nii.gz')
    # utils.io.image.write_np(tf.cast(ret, tf.float32), 'debug/ret.nii.gz')
    return ret


