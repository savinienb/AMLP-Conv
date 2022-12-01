
import tensorflow.compat.v1 as tf
from tensorflow_train.data_generator_queue import DataGeneratorQueue


class DataGenerator(DataGeneratorQueue):
    """
    Basic DataGenerator with a tf.FIFOQueue.
    """
    def init_queue(self):
        """
        Init the queue.
        """
        queue_types = []
        queue_shapes = []
        self.placeholders = []
        for (name, shape) in self.data_names_and_shapes:
            types = self.data_types[name]
            queue_shapes.append([self.batch_size] + shape)
            queue_types.append(types)
            self.placeholders.append(tf.placeholder(types, [self.batch_size] + shape, name='placeholder_' + name))

        self.queue = tf.FIFOQueue(self.queue_size, queue_types, queue_shapes)
        self.enqueue = self.queue.enqueue(self.placeholders)

    def get_feed_dict(self):
        """
        Return the feed_dict that is used in super.thread_main() to feed the placeholders.
        :return: The feed_dict.
        """
        return self.get_feed_dict_batch()
