
import tensorflow.compat.v1 as tf

major, minor, patch = tf.__version__.split('.')
if int(major) == 1 and int(minor) < 10:
    he_initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
    selu_initializer = tf.variance_scaling_initializer(scale=1.0, mode='fan_in', distribution='normal')
else:
    he_initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='truncated_normal')
    selu_initializer = tf.variance_scaling_initializer(scale=1.0, mode='fan_in', distribution='truncated_normal')
zeros_initializer = tf.zeros_initializer
