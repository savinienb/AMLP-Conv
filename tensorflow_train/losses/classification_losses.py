
import tensorflow.compat.v1 as tf


def focal_loss(logits, labels, alpha=0.25, gamma=2):
    """
    Compute focal loss for predictions (adapted from https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/).
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     logits: A float tensor of shape [batch_size, num_classes] representing the predicted logits for each class
     labels: A float tensor of shape [batch_size, num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    assert logits.get_shape().as_list() == labels.get_shape().as_list(), 'Shapes of logits and labels must be the same.'
    sigmoid_p = tf.nn.sigmoid(logits)
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = tf.where(labels > zeros, labels - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(labels > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)
