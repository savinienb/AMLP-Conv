
import tensorflow.compat.v1 as tf
import numpy as np


def heatmap_embedding_loss_single_image(embeddings, landmarks, normalize, sigma, data_format='channels_first'):
    num_landmarks = landmarks.get_shape().as_list()[1]
    landmarks_int = tf.cast(landmarks, tf.int32)
    valid_landmarks = landmarks[:, :, 0] > 0
    valid_landmarks_float = tf.cast(valid_landmarks, tf.float32)
    num_valid_landmarks = tf.reduce_sum(valid_landmarks_float, axis=1)
    valid_persons = num_valid_landmarks > 0
    valid_persons_float = tf.cast(valid_persons, tf.float32)
    num_persons = tf.reduce_sum(valid_persons_float)

    # calculate embeddings per landmark and person -> h_k[person_index, landmark_index]
    h_k_list = []
    for i in range(num_landmarks):
        if data_format == 'channels_first':
            if embeddings.get_shape().as_list()[0] == 1:
                h_k_list.append(tf.gather_nd(embeddings[0, :, :], landmarks_int[:, i, 1:3]))
            else:
                h_k_list.append(tf.gather_nd(embeddings[i, :, :], landmarks_int[:, i, 1:3]))
        elif data_format == 'channels_last':
            if embeddings.get_shape().as_list()[2] == 1:
                h_k_list.append(tf.gather_nd(embeddings[:, :, 0], landmarks_int[:, i, 1:3]))
            else:
                h_k_list.append(tf.gather_nd(embeddings[:, :, i], landmarks_int[:, i, 1:3]))
    h_k = tf.stack(h_k_list, axis=1)
    h_k = tf.where(valid_landmarks, h_k, tf.zeros_like(h_k))

    # calculate mean embedding per person -> h_n[person_index]
    h_n = tf.reduce_sum(h_k, axis=1) / num_valid_landmarks
    h_n = tf.where(valid_persons, h_n, tf.zeros_like(h_n))

    # calculate first term of loss -> deviation of embeddings per landmark and person to mean embedding per person
    h_n_minus_h_k = tf.reshape(h_n, (-1, 1)) - h_k
    term_0_all = (h_n_minus_h_k ** 2)
    term_0_all = tf.where(valid_landmarks, term_0_all, tf.zeros_like(term_0_all))
    term_0 = tf.reduce_sum(term_0_all)
    if normalize:
        normalization_factor = num_persons
        term_0 = tf.cond(normalization_factor > 0, lambda: term_0 / normalization_factor, lambda: 0.0)

    # calculate second term of loss -> deviation of mean embedding per person to mean embedding of every other person
    h_n_times_h_n = tf.reshape(h_n, (-1, 1)) - tf.reshape(h_n, (1, -1))
    h_n_times_h_n_mask = (tf.reshape(valid_persons_float, (-1, 1)) * tf.reshape(valid_persons_float, (1, -1)))
    h_n_times_h_n_mask = tf.matrix_band_part(h_n_times_h_n_mask, 0, -1) > 0
    term_1_all = tf.exp(-(1 / (2 * sigma ** 2)) * (h_n_times_h_n ** 2))
    term_1_all = tf.where(h_n_times_h_n_mask, term_1_all, tf.zeros_like(term_1_all))
    term_1 = tf.reduce_sum(term_1_all)
    if normalize:
        normalization_factor = (num_persons * (num_persons - 1)) / 2
        term_1 = tf.cond(normalization_factor > 0, lambda: term_1 / normalization_factor, lambda: 0.0)

    # final loss is sum over the two terms
    loss = term_0 + term_1
    return loss


def heatmap_embedding_loss(embeddings, landmarks, normalize=False, sigma=0.25, data_format='channels_first'):
    batch_size = landmarks.get_shape().as_list()[0]
    return tf.reduce_mean([heatmap_embedding_loss_single_image(embeddings[i, ...], landmarks[i, ...], normalize, sigma, data_format) for i in range(batch_size)])


def generate_heatmap_target(heatmap_size, landmarks, sigmas, scale=1.0, data_format='channels_first'):
    landmarks_shape = landmarks.get_shape().as_list()
    sigmas_shape = sigmas.get_shape().as_list()
    batch_size = landmarks_shape[0]
    num_landmarks = landmarks_shape[1]
    landmarks_dim = landmarks_shape[2] - 1
    assert len(heatmap_size) == landmarks_dim, 'Dimensions do not match.'
    assert sigmas_shape[0] == num_landmarks, 'Number of sigmas does not match.'
    dim = len(heatmap_size)
    if data_format == 'channels_first':
        heatmap_stack_axis = 0
    else:
        heatmap_stack_axis = dim

    grid = tf.meshgrid(np.arange(heatmap_size[0]), np.arange(heatmap_size[1]), np.arange(heatmap_size[2]), indexing='ij')
    grid_stacked = tf.stack(grid, axis=3)
    grid[0] = tf.cast(grid[0], tf.float32)
    grid[1] = tf.cast(grid[1], tf.float32)
    grid[2] = tf.cast(grid[2], tf.float32)
    grid_stacked = tf.cast(grid_stacked, tf.float32)
    grid_stacked = tf.stack([grid_stacked] * num_landmarks, axis=0)
    grid_stacked = tf.stack([grid_stacked] * batch_size, axis=0)
    zeros = tf.zeros(heatmap_size, tf.float32)
    curr_heatmaps_batch = []
    # for b in range(batch_size):
    #     curr_heatmaps = []
    #     current_image_landmarks = landmarks[b, ...]
    #     for l in range(num_landmarks):
    #         if data_format == 'channels_first':
    #             current_landmark = current_image_landmarks[l, ...]
    #         else:
    #             current_landmark = current_image_landmarks[..., l]
    #         current_landmark_is_valid = current_landmark[0]
    #         current_landmark_coord = current_landmark[1:]
    #         squared_distances = tf.pow(grid[0] - current_landmark_coord[0], 2.0) +\
    #                             tf.pow(grid[1] - current_landmark_coord[1], 2.0) +\
    #                             tf.pow(grid[2] - current_landmark_coord[2], 2.0)
    #         heatmap = scale * tf.exp(-squared_distances / 2 * tf.pow(sigmas[l], 2) / tf.pow(tf.sqrt(2 * np.pi) * sigmas[l], 2))
    #         heatmap_or_zeros = tf.cond(current_landmark_is_valid > 0, lambda: heatmap, lambda: zeros)
    #         curr_heatmaps.append(heatmap_or_zeros)
    #     curr_heatmaps_batch.append(tf.stack(curr_heatmaps, axis=heatmap_stack_axis))
    landmarks_reshaped = tf.reshape(landmarks[..., 1:], [batch_size, num_landmarks, 1, 1, 1, 3])
    squared_distances = tf.reduce_sum(tf.pow(grid_stacked - landmarks_reshaped, 2.0), axis=-1)
    sigmas_reshaped = tf.reshape(sigmas, [batch_size, num_landmarks, 1, 1, 1])
    heatmap = scale * tf.exp(-squared_distances / (2 * tf.pow(sigmas_reshaped, 2)))
    #heatmap_or_zeros = tf.cond(current_landmark_is_valid > 0, lambda: heatmap, lambda: zeros)
    heatmap_or_zeros = tf.where(tf.is_nan(heatmap), tf.zeros_like(heatmap), heatmap)

    return heatmap_or_zeros


def landmark_softmax_loss(predictions, landmarks, data_format='channels_first'):
    predictions_shape = predictions.get_shape().as_list()
    landmarks_shape = landmarks.get_shape().as_list()
    num_landmarks = landmarks_shape[1]

    #assert len(predictions_shape) == len(landmarks_shape), 'Dimensions do not match.'
    assert predictions_shape[1] == landmarks_shape[1], 'Number of landmarks does not match.'

    batch_index = 0
    losses = []
    for i in range(num_landmarks):
        logits = tf.reshape(predictions[batch_index, i, :, :, :], [1, -1])
        coords = tf.cast(tf.round(landmarks[batch_index, i, 1:]), tf.int64)
        labels = tf.cast(tf.reshape(tf.sparse_to_dense([coords], predictions_shape[2:], [1]), [1, -1]), tf.float32)
        current_predictions_softmax = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) * landmarks[batch_index, i, 0]
        losses.append(current_predictions_softmax)

    return tf.reduce_sum(losses)
