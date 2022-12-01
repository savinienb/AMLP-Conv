
from collections import OrderedDict
import sys

import tensorflow as tf
import numpy as np


class DataGeneratorBase(object):
    """
    DataGeneratorBase class that generates np entries from a given dataset, and returns tf.Tensor objects
    that can contain the entries.
    """
    def __init__(self,
                 dataset,
                 data_names_and_shapes,
                 batch_size,
                 data_types=None,
                 queue_size=32):
        """
        Initializer.
        :param dataset: The dataset.
        :param data_names_and_shapes: List or OrderedDict of (name, shape) tuples.
        :param batch_size: The batch size.
        :param data_types: Optional dictionary of data_types (name, tf.DType)
        :param queue_size: The maximum size of the queue.
        """
        assert (sys.version_info.major == 3 and sys.version_info.minor >= 7 and isinstance(data_names_and_shapes, dict)) \
               or isinstance(data_names_and_shapes, OrderedDict) or isinstance(data_names_and_shapes, list), \
               'only OrderedDict and list are allowed for data_names_and_shapes'
        self.dataset = dataset
        if isinstance(data_names_and_shapes, dict) or isinstance(data_names_and_shapes, OrderedDict):
            self.data_names_and_shapes = list(data_names_and_shapes.items())
        elif isinstance(data_names_and_shapes, list):
            self.data_names_and_shapes = data_names_and_shapes
        self.data_types = data_types or {}
        for name, _ in self.data_names_and_shapes:
            if name not in self.data_types:
                self.data_types[name] = tf.float32
        self.batch_size = batch_size
        self.queue_size = queue_size

    def num_entries(self):
        """
        Return the number of dataset entries.
        :return: The number of dataset entries.
        """
        return self.dataset.num_entries()

    def get_queue_types_and_shapes_tuples(self):
        """
        Return tuple of types and shapes for the queue entries.
        :return: Tuple of types, tuple of shapes.
        """
        queue_types = []
        queue_shapes = []
        for (name, shape) in self.data_names_and_shapes:
            types = self.data_types[name]
            queue_shapes.append(shape)
            queue_types.append(types)
        return tuple(queue_types), tuple(queue_shapes)

    def get_tf_data(self, data_generators):
        """
        Return a tuple of tf arrays for a single entry of a batch.
        :return: Tuple of tf arrays for a single entry of a batch.
        """
        tf_data_list = []
        for name, _ in self.data_names_and_shapes:
            tf_data_list.append(tf.convert_to_tensor(data_generators[name], dtype=self.data_types[name]))
        return tuple(tf_data_list)

    def get_empty_tf_data(self):
        """
        Return a tuple of zero tf arrays for a single entry of a batch.
        :return: Tuple of zero tf arrays for a single entry of a batch.
        """
        tf_data_list = {}
        for name, shape in self.data_names_and_shapes:
            tf_shape = [s if s is not None else 1 for s in shape]
            tf_data_list[name] = tf.zeros(tf_shape, dtype=self.data_types[name])
        return tuple(tf_data_list)

    def close(self):
        """
        Optional close operation.
        """
        pass
