

import tensorflow.compat.v1 as tf
from utils.layers import conv3d, upsample3d, max_pool3d

class Unet(object):
    def __init__(self,
                 num_features_per_level,
                 dropout_per_level,
                 activation=tf.nn.relu,
                 use_batch_norm=False):
        self.num_features_per_level = num_features_per_level
        self.dropout_per_level = dropout_per_level
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.is_training = False

    def downsample(self, node, current_level):
        return max_pool3d(node, [2, 2, 2], name='downsample' + str(current_level))

    def upsample(self, node, current_level):
        return upsample3d(node, [2, 2, 2], name='upsample' + str(current_level))

    def combine(self, node0, node1, current_level):
        return tf.concat([node0, node1], axis=1, name='concat' + str(current_level))

    def conv(self, node, current_level, postfix):
        return conv3d(node,
                      [3, 3, 3],
                      self.num_features_per_level[current_level],
                      weight_filler='he_normal',
                      stddev=None,
                      activation=self.activation,
                      use_batch_norm=self.use_batch_norm,
                      is_training=self.is_training,
                      name='conv' + str(current_level) + postfix)

    #def deform_conv(self, node, current_level, postfix):
    #    return deform_conv_2d(node, self.num_features_per_level[current_level], 3, 1, name='deform_conv' + str(current_level) + postfix)

    def dropout(self, node, current_level, postfix):
        dropout_ratio = self.dropout_per_level[current_level]
        if dropout_ratio > 0:
            node = tf.layers.dropout(node, dropout_ratio, training=self.is_training, name='dropout' + str(current_level) + postfix)
        return node

    def contracting_block(self, node, current_level):
        node = self.conv(node, current_level, '_contracting_0')
        node = self.dropout(node, current_level, '_contracting')
        node = self.conv(node, current_level, '_contracting_1')
        return node

    def expanding_block(self, node, current_level):
        node = self.conv(node, current_level, '_expanding_0')
        node = self.dropout(node, current_level, '_expanding')
        node = self.conv(node, current_level, '_expanding_1')
        return node

    def contracting(self, node):
        print('contracting path')
        level_nodes = []
        for current_level in range(len(self.num_features_per_level)):
            node_shape = node.get_shape().as_list()
            print(current_level, node_shape)
            node = self.contracting_block(node, current_level)
            level_nodes.append(node)
            # perform downsampling, if not at last level
            if current_level < len(self.num_features_per_level) - 1:
                node = self.downsample(node, current_level)
        return level_nodes

    def expanding(self, level_nodes):
        print('expanding path')
        node = []
        for current_level in reversed(range(len(self.num_features_per_level))):
            if current_level == len(self.num_features_per_level) - 1:
                # on deepest level, do not combine nodes
                node = level_nodes[current_level]
            else:
                node = self.upsample(node, current_level)
                node = self.combine(level_nodes[current_level], node, current_level)
            node = self.expanding_block(node, current_level)
            node_shape = node.get_shape().as_list()
            print(current_level, node_shape)
        return node

    def net(self, node, is_training=False):
        self.is_training = is_training
        return self.expanding(self.contracting(node))

    def net_with_additional_output(self, node, num_outputs, is_training=False):
        net = self.net(node, is_training)
        return conv3d(net,
                      [3, 3, 3],
                      num_outputs,
                      stddev=0.0001,
                      activation=self.activation,
                      use_batch_norm=self.use_batch_norm,
                      is_training=self.is_training,
                      name='output')

