
import tensorflow as tf


class ContractingNetBase(tf.keras.layers.Layer):
    """
    ContractingNet class as a keras Layer.
    """
    def __init__(self,
                 num_levels,
                 *args, **kwargs):
        """
        Initializer.
        :param num_levels: Number of levels.
        :param args: *args
        :param kwargs: **kwargs
        """
        super(ContractingNetBase, self).__init__(*args, **kwargs)
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

    def contracting_block(self, current_level):
        """
        Create and return the contracting block keras layer for the current level.
        :param current_level: The current level.
        :return: The keras.layers.Layer.
        """
        return None

    def build_contracting(self, input_shape):
        """
        Build the contracting part of the net for the given node.
        :param input_shape: The input shape.
        :return: The output shapes of all levels of the contracting block.
        """
        current_shape = input_shape
        contracting_level_node_shapes = []
        for current_level in range(self.num_levels):
            current_shape = self.build_layer_and_return_output_shape(self.contracting_layers[f'level{current_level}'], current_shape)
            contracting_level_node_shapes.append(current_shape)
            if current_level < self.num_levels - 1:
                current_shape = self.build_layer_and_return_output_shape(self.downsample_layers[f'level{current_level}'], current_shape)
        return contracting_level_node_shapes

    def init_layers(self):
        """
        Create the layers.
        """
        for current_level in range(self.num_levels):
            self.contracting_layers[f'level{current_level}'] = self.contracting_block(current_level)
            if current_level < self.num_levels - 1:
                self.downsample_layers[f'level{current_level}'] = self.downsample(current_level)

    def call_downsample(self, node, current_level, training):
        """
        Call the downsample layer of the current level for the given node.
        :param node: The input node.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        current_layer = self.downsample_layers[f'level{current_level}']
        return node if current_layer is None else current_layer(node, training=training)

    def call_contracting_block(self, node, current_level, training):
        """
        Call the contracting block layer of the current level for the given node.
        :param node: The input node.
        :param current_level: The current level.
        :param training: Training parameter passed to layer.
        :return: The output of the layer.
        """
        if f'level{current_level}' in self.contracting_layers:
            return self.contracting_layers[f'level{current_level}'](node, training=training)
        else:
            return node

    def call_contracting(self, node, training):
        """
        Call the contracting part of the net for the given node.
        :param node: The input node.
        :param training: Training parameter passed to layers.
        :return: The outputs of all levels of the contracting block.
        """
        for current_level in range(self.num_levels):
            node = self.call_contracting_block(node, current_level, training)
            if current_level < self.num_levels - 1:
                node = self.call_downsample(node, current_level, training)
        return node

    def call(self, node, training, **kwargs):
        """
        Call the U-Net for the given input node.
        :param node: The input node.
        :param training: Training parameter passed to layers.
        :param kwargs: **kwargs
        :return: The output of the U-Net.
        """
        return self.call_contracting(node, training)
