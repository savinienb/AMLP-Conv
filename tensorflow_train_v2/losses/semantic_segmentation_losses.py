
import tensorflow as tf
import numpy as np
from tensorflow_train_v2.utils.data_format import get_image_axes, get_channel_index, get_batch_channel_image_size, channels_first_to_channels_last, channels_last_to_channels_first
from tensorflow_train_v2.utils.tensorflow_util import reduce_mean_weighted, reduce_mean_loss_per_pixel


# TODO: clean up and document the file


def softmax_cross_entropy_with_logits(labels, logits, weights=None, data_format='channels_first'):
    # # TODO hack - unstable gradient when computed like this
    # channel_index = get_channel_index(labels, data_format)
    # softmax = tf.nn.softmax(tf.cast(logits, tf.float32))
    # loss_per_pixel = cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=softmax, data_format=data_format)
    # return loss_per_pixel

    # NOTE original
    channel_index = get_channel_index(labels, data_format)
    loss_per_pixel = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.cast(logits, tf.float32), axis=channel_index)
    return reduce_mean_loss_per_pixel(loss_per_pixel, weights, data_format=data_format)

#
# def softmax_cross_entropy_with_logits_weighted(labels, logits, weight_epsilon=1e-8, data_format='channels_first'):
#     channel_index = get_channel_index(labels, data_format)
#     loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, dim=channel_index)
#     image_axes = get_image_axes(labels, data_format)
#     weights = 1 / (tf.reduce_sum(labels, axis=image_axes) + weight_epsilon)
#     batch_size = labels.shape.as_list()[0]
#     indizes = tf.cast(tf.argmax(labels, axis=1), tf.int32)
#     whole_loss = []
#     for i in range(batch_size):
#         batch_weights = tf.gather(weights[i, :], tf.reshape(indizes[i, ...], (-1,)))
#         batch_loss = tf.reshape(loss[i, ...], (-1, ))
#         whole_loss.append(tf.reduce_sum(batch_weights * batch_loss))
#     return tf.reduce_mean(whole_loss)


# def softmax_cross_entropy_with_logits_per_label(labels, logits, weights=None, data_format='channels_first'):
#     #TODO implemented by Franz, loss is computed incorrectly
#     channel_index = get_channel_index(labels, data_format)
#     labels_list = tf.unstack(labels, axis=1)
#     logits_list = tf.unstack(logits, axis=1)
#     loss_list = []
#     for cur_label, cur_logit in zip(labels_list, logits_list):
#         cur_label = tf.expand_dims(cur_label, axis=1)
#         cur_logit = tf.expand_dims(cur_logit, axis=1)
#         cur_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(cur_label, tf.float32), logits=tf.cast(cur_logit, tf.float32), axis=channel_index)
#         # if weights is not None:
#         #     channel_index = get_channel_index(weights, data_format)
#         #     weights = tf.squeeze(weights, axis=channel_index)
#         #     return reduce_mean_weighted(loss, weights)
#         # else:
#         loss_list.append(tf.reduce_sum(cur_loss) / tf.cast(tf.size(labels[0]), dtype=tf.float32))
#     return loss_list


def cross_entropy_with_logits(labels, logits=None, logits_as_probability=None, weights=None, epsilon=1e-8, data_format='channels_first'):
    if logits_as_probability is None:
        logits_as_probability = logits
    channel_index = get_channel_index(labels, data_format)
    image_axes = list(range(1, labels.shape.ndims - 1))
    loss_per_pixel = -tf.reduce_sum(tf.cast(labels, logits_as_probability.dtype) * tf.math.log(logits_as_probability + tf.constant(epsilon, logits_as_probability.dtype)), axis=channel_index)
    return reduce_mean_loss_per_pixel(loss_per_pixel, weights, data_format=data_format)


def normalized_sigmoid(prediction_wo_activation, data_format='channels_first', epsilon=1e-8):
    '''
    This function was added by Franz.
    '''
    channel_axis = 1 if data_format == 'channels_first' else -1
    prediction = tf.nn.sigmoid(prediction_wo_activation)
    normalized_prediction = prediction / (tf.reduce_sum(prediction, axis=channel_axis, keepdims=True) + tf.constant(epsilon, prediction_wo_activation.dtype))
    return normalized_prediction

# def entropy():
#     # Note by Franz: both are equivalent
#     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sigmoid(spatial_prediction), logits=spatial_prediction)
#     -tf.reduce_sum(tf.concat([tf.sigmoid(spatial_prediction), 1 - tf.sigmoid(spatial_prediction)], axis=0) * tf.math.log(tf.concat([tf.sigmoid(spatial_prediction), 1 - tf.sigmoid(spatial_prediction)], axis=0) + tf.constant(1e-8)), axis=0, keepdims=True)


def normalized_sigmoid_cross_entropy_with_logits(labels, logits, weights=None, epsilon=1e-8, data_format='channels_first'):
    '''
    This function was added by Franz.
    '''
    normalized_logits = normalized_sigmoid(logits)
    # normalized_logits = tf.nn.softmax(logits)  # TODO hack
    channel_index = get_channel_index(labels, data_format)
    image_axes = list(range(1, labels.shape.ndims - 1))
    loss_per_pixel = -tf.reduce_sum(tf.cast(labels, normalized_logits.dtype) * tf.math.log(normalized_logits + tf.constant(epsilon, normalized_logits.dtype)), axis=channel_index)
    return reduce_mean_loss_per_pixel(loss_per_pixel, weights, data_format=data_format)


def generalized_dice_loss_normalized_sigmoid(labels, logits=None, data_format='channels_first', weights=None, weight_labels=True, squared=True, weight_epsilon=1e-04, epsilon=1e-08):
    '''
    This function was added by Franz.
    '''
    # assert (logits is None and logits_as_probability is not None) or (logits is not None and logits_as_probability is None), 'Set either logits or logits_as_probability, but not both.'
    dtype = logits.dtype if logits is not None else logits.dtype
    labels = labels if labels.dtype == dtype else tf.cast(labels, dtype)
    channel_index = get_channel_index(labels, data_format)
    image_axes = get_image_axes(labels, data_format)
    labels_shape = labels.get_shape().as_list()
    num_labels = labels_shape[channel_index]
    normalized_logits = normalized_sigmoid(logits)
    # calculate logits propability as softmax (p_n)
    # if logits_as_probability is None:
    #     logits_as_probability = tf.nn.softmax(logits, axis=channel_index)
    if weight_labels:
        # calculate label weights (w_l)
        label_weights = tf.constant(1, dtype=dtype) / (tf.reduce_sum(labels, axis=image_axes) ** 2 + tf.constant(weight_epsilon, dtype=dtype))
    else:
        label_weights = tf.constant(1, dtype=dtype)
    # GDL_b based on equation in reference paper
    numerator = tf.reduce_sum(label_weights * tf.reduce_sum(labels * normalized_logits, axis=image_axes), axis=1)
    if squared:
        # square logits, no need to square labels, as they are either 0 or 1
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + (normalized_logits**2), axis=image_axes), axis=1)
    else:
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + normalized_logits, axis=image_axes), axis=1)
    loss = 1 - 2 * (numerator + epsilon) / (denominator + epsilon)

    if weights is not None:
        channel_index = get_channel_index(weights, data_format)
        weights = tf.squeeze(weights, axis=channel_index)
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def scn_sigmoid_cross_entropy_with_logits(labels, logits_a, logits_b, weights=None, data_format='channels_first'):
    loss = (tf.cast(labels, logits_a.dtype) - 1) * tf.math.log1p(tf.exp(logits_a) + tf.exp(logits_b)) + tf.math.log1p(tf.exp(logits_a)) + tf.math.log1p(tf.exp(logits_b)) - tf.cast(labels, logits_a.dtype) * (logits_a + logits_b)
    if weights is not None:
        channel_index = get_channel_index(weights, data_format)
        weights = tf.squeeze(weights, axis=channel_index)
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def multiclass_hinge_loss(labels, logits, weights=None, data_format='channels_first'):
    channel_index = get_channel_index(labels, data_format)
    labels = 2 * labels - 1
    loss_per_pixel = tf.reduce_sum(tf.nn.relu(1 - (labels * logits)), axis=channel_index)
    return reduce_mean_loss_per_pixel(loss_per_pixel, weights, data_format=data_format)


def sigmoid_cross_entropy_with_logits_wo_sigmoid(labels, logits, weights=None, data_format='channels_first'):
    loss = tf.maximum(logits, 0) - logits * tf.cast(labels, logits.dtype) + tf.log(1 + tf.exp(-tf.abs(logits)))
    if weights is not None:
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def sigmoid_cross_entropy_with_logits(labels, logits, labels_mask=None, weights=None, data_format='channels_first'):
    if labels_mask is not None:
        labels_mask = tf.cast(labels_mask, dtype=labels.dtype)
        labels *= labels_mask
        logits *= labels_mask

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.cast(logits, tf.float32))
    if weights is not None:
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def generalized_dice_loss_with_sigmoid(labels, logits=None, logits_as_probability=None, data_format='channels_first', weights=None, weight_labels=True, squared=True, weight_epsilon=1e-04, epsilon=1e-08):
    """
    Taken from Generalised Dice overlap as a deep learning loss function for
    highly unbalanced segmentations (https://arxiv.org/pdf/1707.03237.pdf)
    :param labels: groundtruth labels
    :param logits: network predictions
    :param data_format: either 'channels_first' of 'channels_last'
    :param epsilon: used for numerical instabilities caused by denominator = 0
    :return: Tensor of mean generalized dice loss of all images of the batch

    This function was added by Franz.
    """
    assert (logits is None and logits_as_probability is not None) or (logits is not None and logits_as_probability is None), 'Set either logits or logits_as_probability, but not both.'
    dtype = logits.dtype if logits is not None else logits_as_probability.dtype
    labels = labels if labels.dtype == dtype else tf.cast(labels, dtype)
    channel_index = get_channel_index(labels, data_format)
    image_axes = get_image_axes(labels, data_format)
    labels_shape = labels.get_shape().as_list()
    num_labels = labels_shape[channel_index]
    # calculate logits propability as softmax (p_n)
    if logits_as_probability is None:
        logits_as_probability = tf.nn.sigmoid(logits)
    if weight_labels:
        # calculate label weights (w_l)
        label_weights = tf.constant(1, dtype=dtype) / (tf.reduce_sum(labels, axis=image_axes) ** 2 + tf.constant(weight_epsilon, dtype=dtype))
    else:
        label_weights = tf.constant(1, dtype=dtype)
    # GDL_b based on equation in reference paper
    numerator = tf.reduce_sum(label_weights * tf.reduce_sum(labels * logits_as_probability, axis=image_axes), axis=1)
    if squared:
        # square logits, no need to square labels, as they are either 0 or 1
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + (logits_as_probability**2), axis=image_axes), axis=1)
    else:
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + logits_as_probability, axis=image_axes), axis=1)
    loss = 1 - 2 * (numerator + epsilon) / (denominator + epsilon)

    if weights is not None:
        channel_index = get_channel_index(weights, data_format)
        weights = tf.squeeze(weights, axis=channel_index)
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def generalized_dice_loss(labels, logits=None, logits_as_probability=None, labels_mask=None, data_format='channels_first', weights=None, weight_labels=True, weight_labels_squared=True, squared=True, weight_epsilon=1e-04, epsilon=1e-08):
    """
    Taken from Generalised Dice overlap as a deep learning loss function for
    highly unbalanced segmentations (https://arxiv.org/pdf/1707.03237.pdf)
    :param labels: groundtruth labels
    :param logits: network predictions
    :param labels_mask: binary mask used to define regions in which the loss is computed (1 = compute loss, 0 = don't compute loss)
    :param data_format: either 'channels_first' of 'channels_last'
    :param epsilon: used for numerical instabilities caused by denominator = 0
    :return: Tensor of mean generalized dice loss of all images of the batch
    """
    assert (logits is None and logits_as_probability is not None) or (logits is not None and logits_as_probability is None), 'Set either logits or logits_as_probability, but not both.'
    dtype = logits.dtype if logits is not None else logits_as_probability.dtype
    labels = labels if labels.dtype == dtype else tf.cast(labels, dtype)
    channel_index = get_channel_index(labels, data_format)
    image_axes = get_image_axes(labels, data_format)
    labels_shape = labels.get_shape().as_list()
    num_labels = labels_shape[channel_index]
    # calculate logits propability as softmax (p_n)
    if logits_as_probability is None:
        logits_as_probability = tf.nn.softmax(logits, axis=channel_index)
    if weight_labels:
        # calculate label weights (w_l)
        if weight_labels_squared:
            label_weights = tf.constant(1, dtype=dtype) / (tf.reduce_sum(labels, axis=image_axes) ** 2 + tf.constant(weight_epsilon, dtype=dtype))
        else:
            # Note: added by Franz to experiment with bcv15-abdomen dataset
            label_weights = tf.constant(1, dtype=dtype) / (tf.reduce_sum(labels, axis=image_axes) + tf.constant(weight_epsilon, dtype=dtype))
    else:
        label_weights = tf.constant(1, dtype=dtype)

    if labels_mask is not None:
        labels_mask = tf.cast(labels_mask, dtype=labels.dtype)
        labels *= labels_mask
        logits_as_probability *= labels_mask

    # GDL_b based on equation in reference paper
    numerator = tf.reduce_sum(label_weights * tf.reduce_sum(labels * logits_as_probability, axis=image_axes), axis=1)
    if squared:
        # square logits, no need to square labels, as they are either 0 or 1
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + (logits_as_probability**2), axis=image_axes), axis=1)
    else:
        denominator = tf.reduce_sum(label_weights * tf.reduce_sum(labels + logits_as_probability, axis=image_axes), axis=1)
    loss = 1 - 2 * (numerator + epsilon) / (denominator + epsilon)

    if weights is not None:
        channel_index = get_channel_index(weights, data_format)
        weights = tf.squeeze(weights, axis=channel_index)
        return reduce_mean_weighted(loss, weights)
    else:
        return tf.reduce_mean(loss)


def wasserstein_disagreement_map(prediction, ground_truth, M):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened pred_proba and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = prediction.get_shape()[1].value
    unstack_labels = tf.unstack(ground_truth, axis=-1)
    unstack_labels = tf.cast(unstack_labels, dtype=tf.float64)
    unstack_pred = tf.unstack(prediction, axis=-1)
    unstack_pred = tf.cast(unstack_pred, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(unstack_pred[i], unstack_labels[j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


def generalised_wasserstein_dice_loss(prediction,
                                      ground_truth,
                                      weight_map=None):
    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in

        Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score
        for Imbalanced Multi-class Segmentation using Holistic
        Convolutional Networks.MICCAI 2017 (BrainLes)

    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    # apply softmax to pred scores
    n_classes = prediction.get_shape()[-1].value
    prediction = tf.reshape(prediction, [-1, n_classes])
    ground_truth = tf.reshape(ground_truth, [-1, n_classes])
    pred_proba = tf.nn.softmax(tf.cast(prediction, dtype=tf.float64))
    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = np.array([[0., 1., 1., 1., 1.],
                  [1., 0., 0.6, 0.2, 0.5],
                  [1., 0.6, 0., 0.6, 0.7],
                  [1., 0.2, 0.6, 0., 0.5],
                  [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)
    # print("M shape is ", M.shape, pred_proba, one_hot)
    delta = wasserstein_disagreement_map(pred_proba, ground_truth, M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(ground_truth, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    return tf.cast(WGDL, dtype=tf.float32)


def softmax_focal_loss(labels, logits=None, logits_as_probability=None, gamma=2.0, alpha=0.25, data_format='channels_first'):
    channel_index = get_channel_index(labels, data_format)
    if logits is None:
        log_softmax = tf.math.log(logits_as_probability)
    else:
        logits_as_probability = tf.nn.softmax(logits, axis=channel_index)
        log_softmax = tf.math.log_softmax(logits, axis=channel_index)

    cross_entropy = - labels * log_softmax
    weight = labels * (1 - logits_as_probability) ** gamma
    focal_loss = alpha * weight * cross_entropy
    focal_loss_per_pixel = tf.math.reduce_sum(focal_loss, axis=channel_index)
    return focal_loss_per_pixel
