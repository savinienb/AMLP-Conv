
import tensorflow.compat.v1 as tf
from collections import OrderedDict


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
        assert isinstance(data_names_and_shapes, OrderedDict) or isinstance(data_names_and_shapes, list), \
            'only OrderedDict and list are allowed for data_names_and_shapes'
        self.dataset = dataset
        if isinstance(data_names_and_shapes, OrderedDict):
            self.data_names_and_shapes = list(data_names_and_shapes.items())
        elif isinstance(data_names_and_shapes, list):
            self.data_names_and_shapes = data_names_and_shapes
        self.data_types = data_types or {}
        for name, _ in self.data_names_and_shapes:
            if name not in self.data_types:
                self.data_types[name] = tf.float32
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.enqueue = None

    def dequeue(self):
        """
        Return the tf tensors.
        :return: tf tensors.
        """
        return NotImplementedError()

    def update(self):
        """
        Return None. Override, to return the update operation.
        :return: None.
        """
        return None

    def start_threads(self, sess):
        """
        Does nothing. Override, if threads should be started.
        :param sess: The current tf.Session object.
        """
        pass

    def close(self, sess):
        """
        Does nothing. Override, if to close the queue and stop threads.
        :param sess: The current tf.Session object.
        """
        pass

    def num_entries(self):
        """
        Return the number of dataset entries.
        :return: The number of dataset entries.
        """
        return self.dataset.num_entries()

    def get_data_batch(self):
        """
        Return a dictionary of np arrays for a whole batch.
        :return: Dictionary of np arrays for a whole batch.
        """
        np_dicts = OrderedDict([(data_name_and_shape[0], []) for data_name_and_shape in self.data_names_and_shapes])
        for _ in range(self.batch_size):
            current_dict = self.dataset.get_next()
            data_generators = current_dict['generators']
            for name, _ in self.data_names_and_shapes:
                np_dicts[name].append(data_generators[name].astype(self.data_types[name].as_numpy_dtype))
        return np_dicts

    def get_data_single(self):
        """
        Return a dictionary of np arrays for a single entry of a batch.
        :return: Dictionary of np arrays for a single entry of a batch.
        """
        np_dicts = OrderedDict()
        current_dict = self.dataset.get_next()
        data_generators = current_dict['generators']
        for name, _ in self.data_names_and_shapes:
            np_dicts[name] = data_generators[name].astype(self.data_types[name].as_numpy_dtype)
        return np_dicts
