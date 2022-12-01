
import tensorflow as tf


class UnetBase(tf.keras.layers.Layer):
    """
    U-Net class as a keras Layer.
    """
    def __init__(self,
                 num_levels,
                 dropout_layer_objects=[],
                 *args, **kwargs):
        """
        Initializer.
        :param num_levels: Number of levels.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(UnetBase, self).__init__(*args, **kwargs)
        self.dropout_layer_objects = dropout_layer_objects
        self.num_levels = num_levels
        self.downsample_layers = {}
        self.upsample_layers = {}
        self.combine_layers = {}
        self.contracting_layers = {}
        self.parallel_layers = {}
        self.expanding_layers = {}

    def downsample(self, current_level):
        """
        Create and return downsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return None

    def upsample(self, current_level):
        """
        Create and return upsample keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return None

    def combine(self, current_level):
        """
        Create and return combine keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return None

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return None

    def parallel_block(self, current_level):
        """
        Create and return the parallel block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return None

    def expanding_block(self, current_level):
        """
        Create and return the expanding block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return None

    def build_contracting(self, input_shape):
        """
        Call the contracting part of the U-Net for the given node.
        :param input_shape: The input shape. Not used.
        :return: The outputs of all levels of the contracting block.
        """
        current_shape = input_shape
        contracting_level_node_shapes = []
        for current_level in range(self.num_levels):
            current_shape = self.layer_output_shape(self.contracting_layers[f'level{current_level}'], current_shape)
            contracting_level_node_shapes.append(current_shape)
            if current_level < self.num_levels - 1:
                current_shape = self.layer_output_shape(self.downsample_layers[f'level{current_level}'], current_shape)
        return contracting_level_node_shapes

    def build_parallel(self, contracting_level_input_shapes):
        """
        Call the contracting part of the U-Net for the given node.
        :param contracting_level_input_shapes: The input node list of the contracting block.
        :return: The outputs of all levels of the parallel block.
        """
        parallel_level_node_shapes = []
        for current_level in range(self.num_levels):
            current_shape = contracting_level_input_shapes[current_level]
            current_shape = self.layer_output_shape(self.parallel_layers[f'level{current_level}'], current_shape)
            parallel_level_node_shapes.append(current_shape)
        return parallel_level_node_shapes

    def build_expanding(self, parallel_level_node_shapes):
        """
        Call the expanding part of the U-Net for the given node.
        :param parallel_level_node_shapes: The input node list of the parallel block.
        :return: The output of the expanding block.
        """
        for current_level in reversed(range(self.num_levels)):
            if current_level == self.num_levels - 1:
                # on deepest level, do not combine nodes
                current_shape = parallel_level_node_shapes[current_level]
            else:
                current_shape = self.layer_output_shape(self.upsample_layers[f'level{current_level}'], current_shape)
                current_shape = self.layer_output_shape(self.combine_layers[f'level{current_level}'], [parallel_level_node_shapes[current_level], current_shape])
            current_shape = self.layer_output_shape(self.expanding_layers[f'level{current_level}'], current_shape)
        return current_shape

    #def build(self, input_shape):
    #    self.build_expanding(self.build_parallel(self.build_contracting(input_shape)))

    def layer_output_shape(self, layer, input_shape):
        if layer is None:
            return input_shape
        layer.build(input_shape)
        if isinstance(input_shape, list):
            output_shape = layer.compute_output_shape([tuple(s.as_list()) for s in input_shape])
        else:
            output_shape = layer.compute_output_shape(tuple(input_shape.as_list()))
        if isinstance(output_shape, list):
            return [tf.TensorShape(s) for s in output_shape]
        else:
            return tf.TensorShape(output_shape)

    def init_layers(self):
        """
        Creates the U-Net layers.
        """
        for current_level in range(self.num_levels):
            self.contracting_layers[f'level{current_level}'] = self.contracting_block(current_level)
            self.parallel_layers[f'level{current_level}'] = self.parallel_block(current_level)
            self.expanding_layers[f'level{current_level}'] = self.expanding_block(current_level)
            if current_level < self.num_levels - 1:
                self.downsample_layers[f'level{current_level}'] = self.downsample(current_level)
                self.upsample_layers[f'level{current_level}'] = self.upsample(current_level)
                self.combine_layers[f'level{current_level}'] = self.combine(current_level)

    def call_downsample(self, node, current_level, training, mc_dropout=False):
        """
        Call the downsample layer of the current level for the given node.
        :param node: The input node.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        current_layer = self.downsample_layers[f'level{current_level}']
        return node if current_layer is None else current_layer(node, training=training)

    def call_upsample(self, node, current_level, training, mc_dropout=False):
        """
        Call the upsample layer of the current level for the given node.
        :param node: The input node.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        current_layer = self.upsample_layers[f'level{current_level}']
        return node if current_layer is None else current_layer(node, training=training)

    def call_combine(self, node_list, current_level, training, mc_dropout=False):
        """
        Call the combine layer of the current level for the given node.
        :param node_list: The input node list to combine.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        current_layer = self.combine_layers[f'level{current_level}']
        return node_list if current_layer is None else current_layer(node_list, training=training)

    def call_contracting_block(self, node, current_level, training, mc_dropout=False):
        """
        Call the contracting block layer of the current level for the given node.
        :param node: The input node.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        current_layer = self.contracting_layers[f'level{current_level}']
        if current_layer is not None:
            for cur_l in current_layer.layers:
                if any([isinstance(cur_l, x) for x in self.dropout_layer_objects]):
                    node = cur_l(node, training=training or mc_dropout)
                else:
                    node = cur_l(node, training=training)
        return node
        # return node if current_layer is None else current_layer(node, training=training)

    def call_parallel_block(self, node, current_level, training, mc_dropout=False):
        """
        Call the parallel block layer of the current level for the given node.
        :param node: The input node.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        current_layer = self.parallel_layers[f'level{current_level}']
        if current_layer is not None:
            for cur_l in current_layer.layers:
                if any([isinstance(cur_l, x) for x in self.dropout_layer_objects]):
                    node = cur_l(node, training=training or mc_dropout)
                else:
                    node = cur_l(node, training=training)
        return node
        # return node if current_layer is None else current_layer(node, training=training)

    def call_expanding_block(self, node, current_level, training, mc_dropout=False):
        """
        Call the expanding block layer of the current level for the given node.
        :param node: The input node.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        current_layer = self.expanding_layers[f'level{current_level}']
        if current_layer is not None:
            for cur_l in current_layer.layers:
                if any([isinstance(cur_l, x) for x in self.dropout_layer_objects]):
                    node = cur_l(node, training=training or mc_dropout)
                else:
                    node = cur_l(node, training=training)
        return node
        # return node if current_layer is None else current_layer(node, training=training)

    def call_contracting(self, node, training, mc_dropout=False):
        """
        Call the contracting part of the U-Net for the given node.
        :param node: The input node.
        :param training: Training parameter passed to layers.
        :return: The outputs of all levels of the contracting block.
        """
        contracting_level_nodes = []
        for current_level in range(self.num_levels):
            node = self.call_contracting_block(node, current_level, training, mc_dropout)
            contracting_level_nodes.append(node)
            if current_level < self.num_levels - 1:
                node = self.call_downsample(node, current_level, training, mc_dropout)
        return contracting_level_nodes

    def call_parallel(self, contracting_level_nodes, training, mc_dropout=False):
        """
        Call the contracting part of the U-Net for the given node.
        :param contracting_level_nodes: The input node list of the contracting block.
        :param training: Training parameter passed to layers.
        :return: The outputs of all levels of the parallel block.
        """
        parallel_level_nodes = []
        for current_level in range(self.num_levels):
            node = self.call_parallel_block(contracting_level_nodes[current_level], current_level, training, mc_dropout)
            parallel_level_nodes.append(node)
        return parallel_level_nodes

    def call_expanding(self, parallel_level_nodes, training, mc_dropout=False):
        """
        Call the expanding part of the U-Net for the given node.
        :param parallel_level_nodes: The input node list of the parallel block.
        :param training: Training parameter passed to layers.
        :return: The output of the expanding block.
        """
        node = None
        for current_level in reversed(range(self.num_levels)):
            if current_level == self.num_levels - 1:
                # on deepest level, do not combine nodes
                node = parallel_level_nodes[current_level]
            else:
                node = self.call_upsample(node, current_level, training, mc_dropout)
                node = self.call_combine([parallel_level_nodes[current_level], node], current_level, training, mc_dropout)
            node = self.call_expanding_block(node, current_level, training, mc_dropout)
        return node

    def call(self, node, training, mc_dropout=False, **kwargs):
        """
        Call the U-Net for the given input node.
        :param node: The input node.
        :param training: Training parameter passed to layers.
        :param kwargs: **kwargs
        :return: The output of the U-Net.
        """
        return self.call_expanding(self.call_parallel(self.call_contracting(node, training, mc_dropout), training, mc_dropout), training, mc_dropout)

