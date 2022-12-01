import tensorflow.compat.v1 as tf



def l1_loss(diff_tensor): #diff_tensor = prediction - target
    abs = tf.abs(diff_tensor)
    mean = tf.reduce_mean(abs)
    return mean