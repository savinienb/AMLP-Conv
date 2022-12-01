
import tensorflow.compat.v1 as tf
from tensorflow.data import Dataset
from tensorflow_train.data_generator_base import DataGeneratorBase
import numpy as np


class DataGeneratorDataset(DataGeneratorBase):
    """
    DataGenerator that is based on the new tf.data input pipeline. Should be faster as other DataGenerators.
    """
    def __init__(self,
                 *args, **kwargs):
        """
        Initializer.
        :param n_threads: The number of background threads. If None, then calculate n_threads based on CPU cores.
        :param args: Arguments, see DataGeneratorBase.
        :param kwargs: Keyword arguments, see DataGeneratorBase.
        """
        # FIXME: convert n_threads to an argument
        if 'n_threads' in kwargs:
            self.n_threads = kwargs.get('n_threads')
            del kwargs['n_threads']
        else:
            self.n_threads = tf.data.experimental.AUTOTUNE
        super(DataGeneratorDataset, self).__init__(*args, **kwargs)
        self.iterator_next = None
        self.init_pipeline()

    def init_pipeline(self):
        """
        Init tf.data pipeline.
        """
        # TODO: very hacky code, as Dataset.from_generator only creates dummy objects, and map does the heavy calculations. Check if this can be changed in newer versions of tf.
        data_pipeline = Dataset.from_generator(self.get_dummy, (tf.string,))
        data_pipeline = data_pipeline.map(self.get_next_pyfunc, num_parallel_calls=self.n_threads)
        data_pipeline = data_pipeline.prefetch(buffer_size=self.queue_size)
        data_pipeline = data_pipeline.batch(batch_size=self.batch_size, drop_remainder=True)
        self.iterator_next = data_pipeline.make_one_shot_iterator().get_next()

    def get_next_pyfunc(self, *args, **kwargs):
        """
        Returns a py_func tensor that generates and returns the next dataset entry.
        :param args: Not used.
        :param kwargs: Not used.
        :return: Tensor tuple with next dataset entry.
        """
        queue_types = []
        queue_shapes = []
        for (name, shape) in self.data_names_and_shapes:
            types = self.data_types[name]
            queue_shapes.append(shape)
            queue_types.append(types)
        entries_wo_shape = tf.py_func(self.get_next, [], queue_types)
        entries = [tf.ensure_shape(entry, shape) for entry, shape in zip(entries_wo_shape, queue_shapes)]
        return tuple(entries)

    def get_dummy(self):
        """
        Return the dummy string tuple as the python generator interface.
        :return: The tuple ('dummy',) as generator.
        """
        while True:
            yield ('dummy',)

    def get_next(self):
        """
        Return the next dataset entry.
        :return: Next dataset entry.
        """
        data_single = self.get_data_single()
        return tuple(data_single.values())


    def get_next_generator(self):
        """
        Return the next dataset entry as the python generator interface.
        :return: Next dataset entry as generator.
        """
        # Dataset.from_generator expects a generator interface
        # this while loop looks weird, but it works as we use yield instead of return -> result matches the python generator interface
        while True:
            data_single = self.get_data_single()
            yield tuple(data_single.values())

    def dequeue(self):
        """
        Return the tf tensors.
        :return: tf tensors.
        """
        return self.iterator_next

