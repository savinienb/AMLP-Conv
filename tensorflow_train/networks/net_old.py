
import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import (conv2d, upsample2d, max_pool2d,
                         conv3d, upsample3d, max_pool3d)
from tensorflow_train.layers.conv_lstm import ConvLSTMCell, ConvGRUCell


def unet_add_recursion(node, current_level, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    input_features = node_shape[1]
    print('down', current_level, node_shape)
    dropout = dropout_per_level[current_level]
    num_features = num_features_per_level[current_level]
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)

    if current_level < max_level:
        node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_2', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        downsample = max_pool2d(node, [2, 2], name='downsample' + str(current_level))
        # recursion
        deeper_level = unet_add_recursion(downsample, current_level + 1, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)
        upsample = upsample2d(deeper_level, [2, 2], name='upsample' + str(current_level))
        #node = upsample + node
        node = tf.add(node, upsample, 'shortcut_add')
        node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_3', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        if dropout > 0:
            node = tf.layers.dropout(node, dropout, training=is_training)

    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_4', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    node_shape = node.get_shape().as_list()
    print('up', current_level, node_shape)
    return node

def unet_add(node, levels, num_features_per_level, variable_scope, dropout_per_level, activation, use_batch_norm=False, is_training=False):
    with tf.variable_scope(variable_scope):
        return unet_add_recursion(node, 0, levels, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)


def unet_add_recursion_deeper(node, current_level, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    input_features = node_shape[1]
    print('down', current_level, node_shape)
    dropout = dropout_per_level[current_level]
    num_features = num_features_per_level[current_level]
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_2', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)

    # deeper level
    if current_level < max_level:
        downsample = max_pool2d(node, [2, 2], name='downsample' + str(current_level))
        # recursion
        deeper_level = unet_add_recursion_deeper(downsample, current_level + 1, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)
        upsample = upsample2d(deeper_level, [2, 2], name='upsample' + str(current_level))
        #node = upsample + node
        node = tf.add(node, upsample, 'shortcut_add')

    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_3', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)

    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_4', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    node_shape = node.get_shape().as_list()
    print('up', current_level, node_shape)
    return node

def unet_add_deeper(node, levels, num_features_per_level, variable_scope, dropout_per_level, activation, use_batch_norm=False, is_training=False):
    with tf.variable_scope(variable_scope):
        return unet_add_recursion_deeper(node, 0, levels, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)

def unet_recursion_deeper(node, current_level, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    input_features = node_shape[1]
    print('down', current_level, node_shape)
    dropout = dropout_per_level[current_level]
    num_features = num_features_per_level[current_level]
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_2', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)

    # deeper level
    if current_level < max_level:
        downsample = max_pool2d(node, [2, 2], name='downsample' + str(current_level))
        # recursion
        deeper_level = unet_add_recursion_deeper(downsample, current_level + 1, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)
        upsample = upsample2d(deeper_level, [2, 2], name='upsample' + str(current_level))
        #node = upsample + node
        node = tf.concat([node, upsample], axis=1, name='concat' + str(current_level))

    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_3', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)

    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_4', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    node_shape = node.get_shape().as_list()
    print('up', current_level, node_shape)
    return node

def unet_deeper(node, levels, num_features_per_level, variable_scope, dropout_per_level, activation, use_batch_norm=False, is_training=False):
    with tf.variable_scope(variable_scope):
        return unet_add_recursion_deeper(node, 0, levels, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)

def unet_add_lstm_recursion(node, lstm_cells, lstm_states, current_level, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    batch_size = node_shape[0]
    input_features = node_shape[1]
    print('down', current_level, node_shape)
    dropout = dropout_per_level[current_level]
    num_features = num_features_per_level[current_level]
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_2', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    node_downsample_input = node

    if len(lstm_cells) <= current_level:
        lstm_cell = ConvLSTMCell(node_shape[2:], num_features, [3, 3], data_format='channels_first', normalize=False, peephole=False, name='lstm' + str(current_level))
        lstm_state = lstm_cell.zero_state(batch_size, tf.float32)
        # data_conv_lstm_output, data_conv_lstm_state = tf.nn.dynamic_rnn(data_conv_lstm, data_conv_concat, dtype=tf.float32)
        node, lstm_state = lstm_cell(node, lstm_state)
        lstm_cells.append(lstm_cell)
        lstm_states.append(lstm_state)
    else:
        lstm_cell = lstm_cells[current_level]
        lstm_state = lstm_states[current_level]
        node, lstm_state = lstm_cell(node, lstm_state)

    # deeper level
    if current_level < max_level:
        downsample = max_pool2d(node_downsample_input, [2, 2], name='downsample' + str(current_level))
        # recursion
        deeper_level, lstm_cells, lstm_states = unet_add_lstm_recursion(downsample, lstm_cells, lstm_states, current_level + 1, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)
        upsample = upsample2d(deeper_level, [2, 2], name='upsample' + str(current_level))
        #node = upsample + node
        node = tf.add(node, upsample, 'shortcut_add')

    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_3', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_4', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    node_shape = node.get_shape().as_list()
    print('up', current_level, node_shape)

    return node, lstm_cells, lstm_states

def unet_add_lstm(node, lstm_cells, lstm_states, levels, num_features_per_level, variable_scope, dropout_per_level, activation, use_batch_norm=False, is_training=False):
    with tf.variable_scope(variable_scope):
        return unet_add_lstm_recursion(node, lstm_cells, lstm_states, 0, levels, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)

def unet_lstm_recursion(node, lstm_cells, lstm_states, current_level, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    batch_size = node_shape[0]
    input_features = node_shape[1]
    print('down', current_level, node_shape)
    dropout = dropout_per_level[current_level]
    num_features = num_features_per_level[current_level]
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_2', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)

    create_lstm = False
    if len(lstm_cells) <= current_level:
        create_lstm = True

    # deeper level
    if current_level < max_level:
        downsample = max_pool2d(node, [2, 2], name='downsample' + str(current_level))
        # recursion
        deeper_level, lstm_cells, lstm_states = unet_lstm_recursion(downsample, lstm_cells, lstm_states, current_level + 1, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)
        upsample = upsample2d(deeper_level, [2, 2], name='upsample' + str(current_level))
        #node = upsample + node
        node = tf.concat([node, upsample], axis=1, name='concat' + str(current_level))

    if create_lstm:
        lstm_cell = ConvLSTMCell(node_shape[2:], num_features, [3, 3], data_format='channels_first', normalize=False, peephole=False, name='lstm' + str(current_level))
        lstm_state = lstm_cell.zero_state(batch_size, tf.float32)
        # data_conv_lstm_output, data_conv_lstm_state = tf.nn.dynamic_rnn(data_conv_lstm, data_conv_concat, dtype=tf.float32)
        node, lstm_state = lstm_cell(node, lstm_state)
        lstm_cells = [lstm_cell] + lstm_cells
        lstm_states = [lstm_state] + lstm_states
    else:
        lstm_cell = lstm_cells[current_level]
        lstm_state = lstm_states[current_level]
        node, lstm_state = lstm_cell(node, lstm_state)

    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_3', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.layers.dropout(node, dropout, training=is_training)
    node = conv2d(node, num_features, [3, 3], name='conv' + str(current_level) + '_4', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    node_shape = node.get_shape().as_list()
    print('up', current_level, node_shape)

    return node, lstm_cells, lstm_states

def unet_lstm(node, lstm_cells, lstm_states, levels, num_features_per_level, variable_scope, dropout_per_level, activation, use_batch_norm=False, is_training=False):
    with tf.variable_scope(variable_scope):
        return unet_lstm_recursion(node, lstm_cells, lstm_states, 0, levels, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)

def unet_3d_symmetric_add_recursion(node, current_level, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    input_features = node_shape[1]
    print('down', current_level, node_shape)
    dropout = dropout_per_level[current_level]
    num_features = num_features_per_level[current_level]
    node = conv3d(tf.pad(node, paddings=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], mode='SYMMETRIC'), num_features, [3, 3, 3], name='conv' + str(current_level) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training, padding='VALID')
    if dropout > 0:
        node = tf.nn.dropout(node, dropout)

    if current_level < max_level:
        node = conv3d(tf.pad(node, paddings=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], mode='SYMMETRIC'), num_features, [3, 3, 3], name='conv' + str(current_level) + '_2', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training, padding='VALID')
        downsample = max_pool3d(node, [1, 2, 2], name='downsample' + str(current_level))
        # recursion
        deeper_level = unet_3d_symmetric_add_recursion(downsample, current_level + 1, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)
        upsample = upsample3d(deeper_level, [1, 2, 2], name='upsample' + str(current_level))
        node = tf.add(node, upsample, 'shortcut_add')
        node = conv3d(tf.pad(node, paddings=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], mode='SYMMETRIC'), num_features, [3, 3, 3], name='conv' + str(current_level) + '_3', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training, padding='VALID')
        if dropout > 0:
            node = tf.nn.dropout(node, dropout)

    node = conv3d(tf.pad(node, paddings=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], mode='SYMMETRIC'), num_features, [3, 3, 3], name='conv' + str(current_level) + '_4', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training, padding='VALID')
    node_shape = node.get_shape().as_list()
    print('up', current_level, node_shape)
    return node


def unet_3d_symmetric_add(node, levels, num_features_per_level, variable_scope, dropout_per_level, activation, use_batch_norm=False, is_training=False):
    with tf.variable_scope(variable_scope):
        return unet_3d_symmetric_add_recursion(node, 0, levels, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)


def unet_3d_add_recursion(node, current_level, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training):
    node_shape = node.get_shape().as_list()
    input_features = node_shape[1]
    print('down', current_level, node_shape)
    dropout = dropout_per_level[current_level]
    num_features = num_features_per_level[current_level]
    node = conv3d(node, num_features, [1, 3, 3], name='conv' + str(current_level) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    if dropout > 0:
        node = tf.nn.dropout(node, dropout)

    if current_level < max_level:
        node = conv3d(node, num_features, [1, 3, 3], name='conv' + str(current_level) + '_2', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        downsample = max_pool3d(node, [1, 2, 2], name='downsample' + str(current_level))
        # recursion
        deeper_level = unet_3d_add_recursion(downsample, current_level + 1, max_level, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)
        upsample = upsample3d(deeper_level, [1, 2, 2], name='upsample' + str(current_level))
        node = tf.add(node, upsample, 'shortcut_add')
        node = conv3d(node, num_features, [1, 3, 3], name='conv' + str(current_level) + '_3', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        if dropout > 0:
            node = tf.nn.dropout(node, dropout)

    print('up', current_level, node_shape)
    return conv3d(node, input_features, [1, 3, 3], name='conv' + str(current_level) + '_4', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)


def unet_3d_add(node, levels, num_features_per_level, variable_scope, dropout_per_level, activation, use_batch_norm=False, is_training=False):
    with tf.variable_scope(variable_scope):
        return unet_3d_add_recursion(node, 0, levels, num_features_per_level, dropout_per_level, activation, use_batch_norm, is_training)


def scnet_intermediate_head_resample_first(data, heatmap_head_mask, num_heatmaps, resample_first_level=2, max_unet_level=5, num_features_data_conv=[128, 128], num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv = data
    for i in range(resample_first_level):
        features_root = num_features_data_conv[i]
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = max_pool2d(data_conv, [2, 2], name='downsample' + str(i))
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv' +  str(i + 1) +'_0')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv' +  str(i + 1) + '_1')
    #data_conv = max_pool2d(data_conv, [2, 2], name='downsample1')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv2_0')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv2_1')

    local_unet = network(data_conv, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv2d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head = conv2d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask, clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local, heatmap_single_head, data_conv], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_unet = network(heatmaps_local_with_single_head_and_image, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person = conv2d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

    return data_conv, heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person


def double_unet_intermediate_head_resample_first(data, heatmap_head_mask, num_heatmaps, resample_first_level=2, max_unet_level=5, num_features_data_conv=[128, 128], num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv = data
    for i in range(resample_first_level):
        features_root = num_features_data_conv[i]
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = max_pool2d(data_conv, [2, 2], name='downsample' + str(i))
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv' +  str(i + 1) +'_0')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv' +  str(i + 1) + '_1')
    #data_conv = max_pool2d(data_conv, [2, 2], name='downsample1')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv2_0')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv2_1')

    local_unet = network(data_conv, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmap_head = conv2d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask, clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmap_single_head_and_image = tf.concat([heatmap_single_head, data_conv], axis=1, name='heatmap_single_head_and_image')

    spatial_unet = network(heatmap_single_head_and_image, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_single_person = conv2d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)

    return data_conv, heatmap_head, heatmaps_single_person


def scnet_intermediate_head_resample_first_with_refinement(data, heatmap_head_mask, num_heatmaps, resample_first_level=2, max_unet_level=5, num_features_data_conv=[128, 128], num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv = data
    for i in range(resample_first_level):
        features_root = num_features_data_conv[i]
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = max_pool2d(data_conv, [2, 2], name='downsample' + str(i))
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv' +  str(i + 1) +'_0')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv' +  str(i + 1) + '_1')
    #data_conv = max_pool2d(data_conv, [2, 2], name='downsample1')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv2_0')
    #data_conv = conv2d(data_conv, kernel_size, features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, name='data_conv2_1')

    local_unet = network(data_conv, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv2d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head = conv2d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask, clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local, heatmap_single_head, data_conv], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_unet = network(heatmaps_local_with_single_head_and_image, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person = conv2d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

    heatmaps_with_image = tf.concat([heatmaps_single_person, data_conv], axis=1, name='heatmaps_with_image')
    refinement_unet = network(heatmaps_with_image, max_unet_level, num_features_per_level, 'refinement_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_single_person_refined = conv2d(refinement_unet, num_heatmaps, kernel_size, name='heatmaps_single_person_refined', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)

    return data_conv, heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person, heatmaps_single_person_refined



def scnet_intermediate_head(data, heatmap_head_mask, num_heatmaps, max_unet_level=5, num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    features_root = num_features_per_level[0]
    kernel_size = [1, 1]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    local_conv0 = conv2d(data, features_root, kernel_size, name='local_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    local_unet = network(local_conv0, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv2d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head = conv2d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask, clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local, heatmap_single_head, data], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_conv0 = conv2d(heatmaps_local_with_single_head_and_image, features_root, kernel_size, name='spatial_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    spatial_unet = network(spatial_conv0, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person = conv2d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

    return data, heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person


def scnet_intermediate_head_resample_first_3d(data, heatmap_head_mask, num_heatmaps, resample_first_level=2, max_unet_level=5, num_features_per_level=[64, 64, 64, 64, 64, 64], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0, head_mask_only_on_first_frame=False, head_mask_only_on_center_frame=False):
    features_root = num_features_per_level[0]
    kernel_size = [1, 3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv = data
    for i in range(resample_first_level):
        data_conv = conv3d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = conv3d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = max_pool3d(data_conv, [1, 2, 2], name='downsample' + str(i))

    local_unet = network(data_conv, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv3d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head = conv3d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.maximum(tf.minimum(heatmap_head_mask, 1.0), heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    if head_mask_only_on_first_frame:
        #heatmap_head_mask_reshape_clamped[:, :, 1:, :, :] = 0
        heatmap_head_mask_shape = heatmap_head_mask_reshape_clamped.get_shape().as_list()
        heatmap_head_mask_shape[2] -= 1
        heatmap_head_mask_reshape_clamped = tf.concat([heatmap_head_mask_reshape_clamped[:, :, 0, None, :, :], tf.zeros(heatmap_head_mask_shape)], axis=2)
        #mask = tf.zeros(heatmap_head_mask_shape, heatmap_head_mask_reshape_clamped.type())
    if head_mask_only_on_center_frame:
        #heatmap_head_mask_reshape_clamped[:, :, 1:, :, :] = 0
        heatmap_head_mask_shape = heatmap_head_mask_reshape_clamped.get_shape().as_list()
        heatmap_head_mask_shape[2] = int(heatmap_head_mask_shape[2] / 2)
        heatmap_head_mask_reshape_clamped = tf.concat([tf.zeros(heatmap_head_mask_shape), heatmap_head_mask_reshape_clamped[:, :, 0, None, :, :], tf.zeros(heatmap_head_mask_shape)], axis=2)
        #mask = tf.zeros(heatmap_head_mask_shape, heatmap_head_mask_reshape_clamped.type())
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local, heatmap_single_head, data_conv], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_conv0 = conv3d(heatmaps_local_with_single_head_and_image, features_root, kernel_size, name='spatial_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    spatial_unet = network(spatial_conv0, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person = conv3d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

    return data_conv, heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person

def scnet_intermediate_head_resample_first_3d_then_2d(data, heatmap_head_mask, num_heatmaps, resample_first_level=2, max_unet_level=5, num_features_per_level=[64, 64, 64, 64, 64, 64], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    features_root = num_features_per_level[0]
    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv = data
    for i in range(resample_first_level):
        data_conv = conv3d(data_conv, features_root, [1, 3, 3], name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        #data_conv = conv3d(data_conv, [3, 1, 1], features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, padding='VALID', name='data_conv' +  str(i) + '_0_reduce_z')
        data_conv = conv3d(data_conv, features_root, [3, 1, 1], name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training, padding='VALID')
        #data_conv = conv3d(data_conv, [3, 1, 1], features_root, weight_filler='he_normal', stddev=None, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training, padding='VALID', name='data_conv' + str(i) + '_1_reduce_z')
        data_conv = max_pool3d(data_conv, [1, 2, 2], name='downsample' + str(i))

    data_conv = data_conv[:, :, 0, :, :]

    local_unet = network(data_conv, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv2d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head = conv2d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask, clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_head_mask_reshape_clamped = heatmap_head_mask_reshape_clamped[:, :, 0, :, :]
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local, heatmap_single_head, data_conv], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_unet = network(heatmaps_local_with_single_head_and_image, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person = conv2d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

    return data_conv, heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person


def scnet_intermediate_head_3d(data, heatmap_head_mask, num_heatmaps, max_unet_level=5, num_features_per_level=[64, 64, 64, 64, 64, 64], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    features_root = num_features_per_level[0]
    kernel_size = [1, 1, 1]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    local_conv0 = conv3d(data, features_root, kernel_size, name='local_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    local_unet = network(local_conv0, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv3d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head = conv3d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.maximum(tf.minimum(heatmap_head_mask, 1.0), heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local, heatmap_single_head, data], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_conv0 = conv3d(heatmaps_local_with_single_head_and_image, features_root, kernel_size, name='spatial_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    spatial_unet = network(spatial_conv0, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person = conv3d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

    return data, heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person


def scnet_volumetric_intermediate_fusion(data, heatmap_head_mask, num_heatmaps, unet_levels=5, features_root=32, heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network_2d=unet_add, network_3d=unet_3d_symmetric_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    data_shape = data.get_shape().as_list()
    batch_size = data_shape[0]
    if batch_size != 1:
        exit(-1)

    data_squeezed = data[0, :, :, :, :]
    heatmap_head_mask_squeezed = heatmap_head_mask[0, :, :, :, :]

    activation = tf.nn.relu  # tf.contrib.keras.layers.LeakyReLU(alpha=0.1)
    local_conv0 = conv2d(data_squeezed, features_root, [1, 1], name='local_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    local_unet = network_2d(local_conv0, unet_levels, features_root, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local_squeezed = conv2d(local_unet, num_heatmaps, [1, 1], name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_local_fused = network_3d(heatmaps_local_squeezed[None, :, :, :, :], 5, 16, 'local_fusion', dropout_per_level=[0, 0, 0, 0, 0, 0], use_batch_norm=use_batch_norm, is_training=is_training )
    heatmap_head_squeezed = conv2d(local_unet, 1, [1, 1], name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_fused = network_3d(heatmap_head_squeezed[None, :, :, :, :], 5, 16, 'head_fusion', dropout_per_level=[0, 0, 0, 0, 0, 0], activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.maximum(tf.minimum(heatmap_head_mask, 1.0), heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head_fused, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local_fused, heatmap_single_head, data], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_conv0 = conv2d(heatmaps_local_with_single_head_and_image[0, :, :, :, :], features_root, [1, 1], name='spatial_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    spatial_unet = network_2d(spatial_conv0, unet_levels, features_root, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person_squeezed = conv2d(spatial_unet, num_heatmaps, [1, 1], name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_spatial_single_person_fused = network_3d(heatmaps_spatial_single_person_squeezed[None, :, :, :, :], 5, 16, 'spatial_fusion', dropout_per_level=[0, 0, 0, 0, 0, 0], use_batch_norm=use_batch_norm, is_training=is_training )
    heatmaps_single_person = tf.multiply(heatmaps_local_fused, heatmaps_spatial_single_person_fused, 'heatmaps')

    return heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person

    heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person = scnet_intermediate_head(tf.squeeze(data, axis=0),
                            tf.squeeze(heatmap_head_mask, axis=0),
                            num_heatmaps,
                            unet_levels=unet_levels,
                            features_root=features_root,
                            heatmap_activation=heatmap_activation,
                            heatmap_kernel_stddev=heatmap_kernel_stddev,
                            network=network_2d, dropout_per_level=dropout_per_level,
                            use_batch_norm=use_batch_norm, is_training=is_training,
                            heatmap_mask_min_value=heatmap_mask_min_value)

    kernel_size = [1, 1, 1]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    local_conv0 = conv3d(data, features_root, kernel_size, name='local_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    local_unet = unet(local_conv0, unet_levels, features_root, 'local_unet', dropout_per_level=dropout_per_level, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv3d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head = conv3d(local_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.maximum(tf.minimum(heatmap_head_mask, 1.0), heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmaps_local_with_single_head_and_image = tf.concat([heatmaps_local, heatmap_single_head, data], axis=1, name='heatmaps_local_with_single_head_and_image')

    spatial_conv0 = conv3d(heatmaps_local_with_single_head_and_image, features_root, kernel_size, name='spatial_conv0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
    spatial_unet = network(spatial_conv0, unet_levels, features_root, 'spatial_unet', dropout_per_level=dropout_per_level, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial_single_person = conv3d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person = tf.multiply(heatmaps_local, heatmaps_spatial_single_person, 'heatmaps')

    return heatmaps_local, heatmaps_spatial_single_person, heatmap_head, heatmaps_single_person


def scnet_double_u_resample_first(data, num_heatmaps, resample_first_level=0, max_unet_level=5, num_features_data_conv=[128, 128], num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False):

    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv = data
    for i in range(resample_first_level):
        features_root = num_features_data_conv[i]
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = max_pool2d(data_conv, [2, 2], name='downsample' + str(i))

    local_unet = network(data_conv, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_local = conv2d(local_unet, num_heatmaps, kernel_size, name='heatmaps_local', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_local_with_image = tf.concat([heatmaps_local, data_conv], axis=1, name='heatmaps_local_with_image')

    spatial_unet = network(heatmaps_local_with_image, max_unet_level, num_features_per_level, 'spatial_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps_spatial = conv2d(spatial_unet, num_heatmaps, kernel_size, name='heatmaps_spatial', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps = tf.multiply(heatmaps_local, heatmaps_spatial, 'heatmaps')

    return data_conv, heatmaps_local, heatmaps_spatial, heatmaps


def unet_resample_first(data, num_heatmaps, resample_first_level=0, max_unet_level=5, num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False):
    features_root = num_features_per_level[0]
    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv = data
    for i in range(resample_first_level):
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = conv2d(data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
        data_conv = max_pool2d(data_conv, [2, 2], name='downsample' + str(i))

    local_unet = network(data_conv, max_unet_level, num_features_per_level, 'local_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmaps = conv2d(local_unet, num_heatmaps, kernel_size, name='heatmaps', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)

    return data_conv, heatmaps


def tracknet_intermediate_head_pose_resample_first(data, heatmap_head_mask, num_heatmaps, resample_first_level=2, max_unet_level=5, num_features_data_conv=[128, 128], num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    data_shape = data.get_shape().as_list()
    print(data_shape)
    num_frames = data_shape[2]

    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv_list = []
    heatmaps_single_person_list = []

    for i in range(num_frames):
        if i == 0:
            reuse = False
        else:
            reuse = True
        with tf.variable_scope('data_conv', reuse=reuse):
            current_data_conv = data[:, :, i, :, :]
            for i in range(resample_first_level):
                features_root = num_features_data_conv[i]
                current_data_conv = conv2d(current_data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
                current_data_conv = conv2d(current_data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
                current_data_conv = max_pool2d(current_data_conv, [2, 2], name='downsample' + str(i))
            data_conv_list.append(current_data_conv)

    head_unet = network(data_conv_list[0], max_unet_level, num_features_per_level, 'head_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmap_head = conv2d(head_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask[:, :, 0, :, :], clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmap_single_head_and_image = tf.concat([heatmap_single_head, data_conv_list[0]], axis=1, name='heatmap_single_head_and_image')
    first_single_net = network(heatmap_single_head_and_image, max_unet_level, num_features_per_level, 'first_single_net', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    first_heatmaps_single_person = conv2d(first_single_net, num_heatmaps, kernel_size, name='first_heatmaps_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person_list.append(first_heatmaps_single_person)

    for i in range(1, num_frames):
        if i == 1:
            reuse = False
        else:
            reuse = True
        with tf.variable_scope('single_net', reuse=reuse):
            heatmaps_single_person_and_image = tf.concat([heatmaps_single_person_list[i - 1], data_conv_list[i - 1], data_conv_list[i]], axis=1, name='heatmaps_single_person_and_images')
            single_net = network(heatmaps_single_person_and_image, max_unet_level, num_features_per_level, 'single_net', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
            current_heatmaps_single_person = conv2d(single_net, num_heatmaps, kernel_size, name='heatmaps_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
            heatmaps_single_person_list.append(current_heatmaps_single_person)

    data_conv_concat = tf.stack(data_conv_list, axis=2, name='data_conv_concat')
    heatmaps_single_person_concat = tf.stack(heatmaps_single_person_list, axis=2, name='heatmaps_single_person_concat')

    return data_conv_concat, heatmap_head, heatmaps_single_person_concat


def tracknet_intermediate_head_pose_resample_first_lstm(data, heatmap_head_mask, num_heatmaps, resample_first_level=2, max_unet_level=5, num_features_data_conv=[128, 128], num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, network=unet_add, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    data_shape = data.get_shape().as_list()
    print(data_shape)
    batch_size = data_shape[0]
    num_frames = data_shape[2]

    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv_list = []
    heatmaps_single_person_list = []

    for i in range(num_frames):
        if i == 0:
            reuse = False
        else:
            reuse = True
        with tf.variable_scope('data_conv', reuse=reuse):
            current_data_conv = data[:, :, i, :, :]
            for i in range(resample_first_level):
                features_root = num_features_data_conv[i]
                current_data_conv = conv2d(current_data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_0', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
                current_data_conv = conv2d(current_data_conv, features_root, kernel_size, name='data_conv' + str(i) + '_1', activation=activation, weight_filler='he_normal', stddev=None, use_batch_norm=use_batch_norm, is_training=is_training)
                current_data_conv = max_pool2d(current_data_conv, [2, 2], name='downsample' + str(i))
            data_conv_list.append(current_data_conv)

    data_conv_shape = data_conv_list[0].get_shape().as_list()
    data_conv_lstm = ConvLSTMCell(data_conv_shape[2:], 128, [3, 3], data_format='channels_first', normalize=False, peephole=False)
    data_conv_lstm_state = data_conv_lstm.zero_state(batch_size, tf.float32)
    #data_conv_lstm_output, data_conv_lstm_state = tf.nn.dynamic_rnn(data_conv_lstm, data_conv_concat, dtype=tf.float32)
    current_data_conv_lstm_output, data_conv_lstm_state = data_conv_lstm(data_conv_list[0], data_conv_lstm_state)

    head_unet = network(current_data_conv_lstm_output, max_unet_level, num_features_per_level, 'head_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    heatmap_head = conv2d(head_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask[:, :, 0, :, :], clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
    heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
    heatmap_single_head_and_image = tf.concat([heatmap_single_head, current_data_conv_lstm_output], axis=1, name='heatmap_single_head_and_image')
    first_single_net = network(heatmap_single_head_and_image, max_unet_level, num_features_per_level, 'first_single_net', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
    first_heatmaps_single_person = conv2d(first_single_net, num_heatmaps, kernel_size, name='first_heatmaps_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
    heatmaps_single_person_list.append(first_heatmaps_single_person)

    heatmaps_single_person_shape = heatmaps_single_person_list[0].get_shape().as_list()
    heatmaps_single_person_lstm = ConvLSTMCell(heatmaps_single_person_shape[2:], 15, [3, 3], data_format='channels_first', normalize=False, peephole=False)
    heatmaps_single_person_lstm_state = data_conv_lstm.zero_state(batch_size, tf.float32)
    #data_conv_lstm_output, data_conv_lstm_state = tf.nn.dynamic_rnn(data_conv_lstm, data_conv_concat, dtype=tf.float32)
    current_heatmaps_single_person_lstm_output, heatmaps_single_person_lstm_state = heatmaps_single_person_lstm(data_conv_list[0], data_conv_lstm_state)

    for i in range(1, num_frames):
        if i == 1:
            reuse = False
        else:
            reuse = True
        with tf.variable_scope('single_net', reuse=reuse):
            current_data_conv_lstm_output, data_conv_lstm_state = data_conv_lstm(data_conv_list[i], data_conv_lstm_state)
            #data_conv_lstm_output, data_conv_lstm_state = data_conv_lstm(tf.contrib.layers.flatten(data_conv_list[i]), data_conv_lstm_state)
            #data_conv_lstm_output = tf.reshape(data_conv_lstm_output, data_conv_shape)
            heatmaps_single_person_and_image = tf.concat([heatmaps_single_person_list[i - 1], current_data_conv_lstm_output, data_conv_list[i]], axis=1, name='heatmaps_single_person_and_images')
            single_net = network(heatmaps_single_person_and_image, max_unet_level, num_features_per_level, 'single_net', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
            heatmaps_single_person = conv2d(single_net, num_heatmaps, kernel_size, name='heatmaps_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
            heatmaps_single_person_list.append(heatmaps_single_person)

    data_conv_concat = tf.stack(data_conv_list, axis=2, name='data_conv_concat')
    heatmaps_single_person_concat = tf.stack(heatmaps_single_person_list, axis=2, name='heatmaps_single_person_concat')

    return data_conv_concat, heatmap_head, heatmaps_single_person_concat


def tracknet_intermediate_head_pose_resample_first_unet_lstm(data, heatmap_head_mask, num_heatmaps, max_unet_level=5, num_features_data_conv=[128, 128], num_features_per_level=[128, 128, 128, 128, 128, 128], heatmap_activation=tf.nn.tanh, heatmap_kernel_stddev=0.01, dropout_per_level=None, use_batch_norm=False, is_training=False, heatmap_mask_min_value=0.0):
    data_shape = data.get_shape().as_list()
    print(data_shape)
    batch_size = data_shape[0]
    num_frames = data_shape[2]

    kernel_size = [3, 3]
    activation = tf.nn.relu #tf.contrib.keras.layers.LeakyReLU(alpha=0.1)

    data_conv_list = []
    heatmaps_single_person_list = []

    for i in range(num_frames):
        if i == 0:
            reuse = False
        else:
            reuse = True
        with tf.variable_scope('data_conv', reuse=reuse):
            current_data_conv = resample_net(data[:, :, i, :, :], num_features_data_conv, use_batch_norm, is_training)
            data_conv_list.append(current_data_conv)

    with tf.variable_scope('first_frame_net'):
        current_data_conv = data_conv_list[0]
        head_unet = unet_deeper(current_data_conv, max_unet_level, num_features_per_level, 'head_unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
        heatmap_head = conv2d(head_unet, 1, kernel_size, name='heatmap_head', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
        heatmap_head_mask_reshape_clamped = tf.clip_by_value(heatmap_head_mask[:, :, 0, :, :], clip_value_max=1.0, clip_value_min=heatmap_mask_min_value, name='heatmap_head_mask_reshape_clamped')
        heatmap_single_head = tf.multiply(heatmap_head, heatmap_head_mask_reshape_clamped, 'heatmap_single_head')
        heatmap_single_head_and_image = tf.concat([heatmap_single_head, current_data_conv], axis=1, name='heatmap_single_head_and_image')
        first_single_net = unet_deeper(heatmap_single_head_and_image, max_unet_level, num_features_per_level, 'unet', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
        first_heatmaps_single_person = conv2d(first_single_net, num_heatmaps, kernel_size, name='heatmaps_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
        heatmaps_single_person_list.append(first_heatmaps_single_person)

    lstm_cells = []
    lstm_states = []
    for i in range(0, num_frames):
        if i == 0:
            reuse = False
        else:
            reuse = True
        with tf.variable_scope('tracking_net', reuse=reuse):
            previous_heatmaps_single_person = heatmaps_single_person_list[i]
            current_data_conv = data_conv_list[i]
            #data_conv_lstm_output, data_conv_lstm_state = data_conv_lstm(tf.contrib.layers.flatten(data_conv_list[i]), data_conv_lstm_state)
            #data_conv_lstm_output = tf.reshape(data_conv_lstm_output, data_conv_shape)
            heatmaps_single_person_and_image = tf.concat([previous_heatmaps_single_person, current_data_conv], axis=1, name='heatmaps_single_person_and_image')
            single_net, lstm_cells, lstm_states = unet_lstm(heatmaps_single_person_and_image, lstm_cells, lstm_states, max_unet_level, num_features_per_level, 'unet_lstm', dropout_per_level=dropout_per_level, activation=activation, use_batch_norm=use_batch_norm, is_training=is_training)
            heatmaps_single_person = conv2d(single_net, num_heatmaps, kernel_size, name='heatmaps_single_person', activation=heatmap_activation, weight_filler=None, stddev=heatmap_kernel_stddev, use_batch_norm=False, is_training=is_training)
            heatmaps_single_person_list.append(heatmaps_single_person)

    data_conv_concat = tf.stack(data_conv_list, axis=2, name='data_conv_concat')
    heatmaps_single_person_concat = tf.stack(heatmaps_single_person_list[1:], axis=2, name='heatmaps_single_person_concat')

    return data_conv_concat, heatmap_head, heatmaps_single_person_concat, heatmaps_single_person_list[0]
