
import tensorflow as tf

from datasets.dataset_queue import DatasetQueue
from tensorflow_train_v2.dataset.data_generator_base import DataGeneratorBase


class DatasetIteratorMultiprocessing(DataGeneratorBase):
    """
    DataGenerator that uses a DatasetQueue to prefetch objects in the background.
    """
    def __init__(self,
                 n_threads=None,
                 prefetch_to_device='/device:GPU:0',
                 use_multiprocessing=True,
                 use_shared_memory_cache=True,
                 use_shared_memory_queue=False,
                 use_queue_cache_compression=True,
                 *args, **kwargs):
        """
        Initializer.
        :param n_threads: The number of background threads. If None, then calculate n_threads based on CPU cores.
        :param prefetch_to_device: The device to copy the data to. '/device:GPU:0' should be save to use. If None, the
                                   prefetched data will be copied to the device not until being accessed.
        :param use_multiprocessing: If True, use multiprocessing for workers, otherwise use threads.
                                    Multiprocessing should be much faster, but will duplicate memory.
        :param use_shared_memory_cache: If True, use shared memory cache in internal dataset_queue.
        :param use_shared_memory_queue: If True, use shared memory queue in internal dataset_queue.
        :param args: Arguments, see DataGeneratorBase.
        :param kwargs: Keyword arguments, see DataGeneratorBase.
        """
        super(DatasetIteratorMultiprocessing, self).__init__(*args, **kwargs)
        self.n_threads = n_threads or tf.data.AUTOTUNE
        self.prefetch_to_device = prefetch_to_device
        # NOTE: although the variable 'running' could be accessed by multiple threads (inside get_next_generator()),
        # it is not secured by a mutex, as it is only set at the initialization and unset at closing.
        self.running = True
        dataset_queue_parameters = {'dataset': self.dataset,
                                    'queue_size': self.queue_size,
                                    'n_threads': self.n_threads,
                                    'use_multiprocessing': use_multiprocessing,
                                    'use_shared_memory_cache': use_shared_memory_cache,
                                    'use_shared_memory_queue': use_shared_memory_queue}
        if not use_queue_cache_compression:
            dataset_queue_parameters['cache_compression_type'] = None
        # init dataset_queue, which will also start the background prefetching threads.
        self.dataset_queue = DatasetQueue(**dataset_queue_parameters)
        self.iterator_next = None
        self.init_pipeline()

    def init_pipeline(self):
        """
        Init tf.data pipeline.
        """
        queue_types, queue_shapes = self.get_queue_types_and_shapes_tuples()
        data_pipeline = tf.data.Dataset.from_generator(self.get_next_generator, queue_types, queue_shapes)
        data_pipeline = data_pipeline.batch(batch_size=self.batch_size, drop_remainder=True)
        data_pipeline.prefetch(buffer_size=tf.data.AUTOTUNE)
        if self.prefetch_to_device:
            data_pipeline = data_pipeline.apply(tf.data.experimental.prefetch_to_device(self.prefetch_to_device, buffer_size=tf.data.AUTOTUNE))
        self.iterator_next = iter(data_pipeline)

    def get_next_generator(self):
        """
        Return the next data entry from the queue as the python generator interface.
        :return: The data entry tuple as generator.
        """
        while self.running:
            error, done, dataset_entry = self.dataset_queue.get()
            if error:
                raise error
            elif done:
                break
            yield self.get_tf_data(dataset_entry['generators'])

    def get_next(self):
        """
        Return the next entry of the iterable object.
        :return: Tuple of the next entry.
        """
        return next(self.iterator_next)

    def close(self):
        """
        Stop the iterator by ending the data generation loop and clearing the queue.
        """
        print('DatasetIteratorMultiprocessing close')
        self.dataset_queue.close()
        self.running = False
        try:
            while True:
                self.get_next()
        except StopIteration:
            pass
        print('DatasetIteratorMultiprocessing closed')
