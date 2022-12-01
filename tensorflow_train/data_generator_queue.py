
from tensorflow_train.data_generator_base import DataGeneratorBase
import threading
import numpy as np


class DataGeneratorQueue(DataGeneratorBase):
    """
    A DataGenerator that uses tf.queue to feed the data. Performs prefetching in background threads.
    This class also only serves as a base class for DataGenerators. Override init_queue() and get_feed_dict().
    """
    def __init__(self,
                 dataset,
                 coord,
                 data_names_and_shapes,
                 batch_size,
                 data_types=None,
                 queue_size=32,
                 n_threads=8):
        """
        Initializer.
        :param dataset: The dataset.
        :param coord: The current tf coordinator.
        :param data_names_and_shapes: List or OrderedDict of (name, shape) tuples.
        :param batch_size: The batch size.
        :param data_types: Optional dictionary of data_types (name, tf.DType)
        :param queue_size: The maximum size of the queue.
        :param n_threads: The number of prefetching threads.
        """
        super(DataGeneratorQueue, self).__init__(dataset=dataset, data_names_and_shapes=data_names_and_shapes, batch_size=batch_size, data_types=data_types, queue_size=queue_size)
        self.n_threads = n_threads
        self.coord = coord
        self.threads = []
        self.placeholders = None
        self.queue = None
        self.enqueue = None
        self.init_queue()

    def init_queue(self):
        """
        Initalizes the queue objects.
        """
        raise NotImplementedError()

    def get_feed_dict(self):
        """
        Return the feed_dict that is used in super.thread_main() to feed the placeholders.
        :return: The feed_dict.
        """
        return NotImplementedError()

    def size(self):
        """
        Return a tensor that holds the current queue size.
        :return: Tensor of the current queue size.
        """
        return self.queue.size()

    def dequeue(self):
        """
        Return the tf tensors.
        :return: tf tensors.
        """
        return self.queue.dequeue()

    def get_feed_dict_batch(self):
        """
        Return the feed_dict for a whole batch.
        :return: The feed_dict for a whole batch.
        """
        np_dicts = self.get_data_batch()
        feed_dict = {}
        for i in range(len(self.data_names_and_shapes)):
            placeholder = self.placeholders[i]
            name = self.data_names_and_shapes[i][0]
            max_shape = np.max([a.shape for a in np_dicts[name]], axis=0)
            padded_values = []
            for a in np_dicts[name]:
                shape = a.shape
                padding = list(zip([0] * len(shape), (max_shape - shape)))
                padded_values.append(np.pad(a, padding, 'constant'))
            feed_dict[placeholder] = np.stack(padded_values)

        return feed_dict

    def get_feed_dict_single(self):
        """
        Return the feed_dict for a single entry of a batch.
        :return: The feed_dict for a single entry of a batch.
        """
        np_dicts = self.get_data_single()
        feed_dict = {}
        for i in range(len(self.data_names_and_shapes)):
            placeholder = self.placeholders[i]
            name = self.data_names_and_shapes[i][0]
            feed_dict[placeholder] = np_dicts[name]

        return feed_dict

    def thread_main(self, sess):
        """
        Main method for a thread.
        :param sess: The current tf.Session object.
        """
        print('Data generator thread start')
        while not self.coord.should_stop():
            try:
                feed_dict = self.get_feed_dict()
                sess.run(self.enqueue, feed_dict=feed_dict)
            except Exception as e:
                # request stop, when there was an exception, but the threads should keep running
                if not self.coord.should_stop():
                    self.coord.request_stop(e)
                    self.close(sess)
                # otherwise, continue loop
                continue
        print('Data generator thread stop')

    def start_threads(self, sess):
        """
        Start the prefetching threads.
        :param sess: The current tf.Session object.
        :return: The list of threads.
        """
        for _ in range(self.n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            self.coord.register_thread(thread)
            thread.start()
            self.threads.append(thread)
        return self.threads

    def close(self, sess):
        """
        Close the queue, stop the threads.
        :param sess: The current tf.Session object.
        """
        sess.run(self.queue.close(True))
