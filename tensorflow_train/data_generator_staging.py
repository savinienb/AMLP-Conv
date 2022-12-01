
import tensorflow.compat.v1 as tf
from tensorflow_train.data_generator_queue import DataGeneratorQueue


class DataGeneratorStaging(DataGeneratorQueue):
    """
    DataGenerator with staging area. The staging area copies data to the gpu when update() is called.
    If dequeue() and update() are called in the same sess.run(), there will be always data on the gpu.
    This should improve performance, but uses slightly more gpu memory.
    """
    def init_queue(self):
        """
        Init the queue with a staging area.
        """
        self.queue = tf.FIFOQueue(self.queue_size,
                                  [self.data_types.get(name, tf.float32) for (name, shape) in self.data_names_and_shapes],
                                  [[self.batch_size] + shape for (name, shape) in self.data_names_and_shapes])
        self.placeholders = [tf.placeholder(self.data_types.get(name, tf.float32), [self.batch_size] + shape, name='placeholder_' + name) for (name, shape) in self.data_names_and_shapes]
        self.enqueue = self.queue.enqueue(self.placeholders)
        self.staging_area = tf.contrib.staging.StagingArea([self.data_types.get(name, tf.float32) for (name, shape) in self.data_names_and_shapes],
                                                           [[self.batch_size] + shape for (name, shape) in self.data_names_and_shapes])
        self.put_op = self.staging_area.put(self.queue.dequeue())
        self.get_op = self.staging_area.get()

    def get_feed_dict(self):
        """
        Return the feed_dict that is used in super.thread_main() to feed the placeholders.
        :return: The feed_dict.
        """
        return self.get_feed_dict_batch()

    def dequeue(self):
        """
        Return the tf tensors.
        :return: tf tensors.
        """
        return self.get_op

    def update(self):
        return self.put_op

    def start_threads(self, sess):
        """
        At first, add a single entry to the staging area. Then, start the prefetching threads.
        :param sess: The current tf.Session object.
        :return: The list of threads.
        """
        feed_dict = self.get_feed_dict()
        sess.run((self.put_op, self.enqueue), feed_dict=feed_dict)
        return super(DataGeneratorStaging, self).start_threads(sess)
