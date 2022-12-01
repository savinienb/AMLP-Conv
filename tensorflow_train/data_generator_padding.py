
import tensorflow.compat.v1 as tf
from tensorflow_train.data_generator_queue import DataGeneratorQueue


class DataGeneratorPadding(DataGeneratorQueue):
    """
    DataGenerator that supports shapes where some entries are None.
    """
    def init_queue(self):
        """
        Init the queue.
        """
        queue_types = []
        queue_shapes = []
        self.placeholders = []
        for (name, shape) in self.data_names_and_shapes:
            if self.data_types is not None and name in self.data_types:
                types = self.data_types[name]
            else:
                types = tf.float32
            queue_shapes.append(shape)
            queue_types.append(types)
            self.placeholders.append(tf.placeholder(types, shape, name='placeholder_' + name))

        self.queue = tf.PaddingFIFOQueue(self.queue_size, queue_types, queue_shapes)
        self.enqueue = self.queue.enqueue(self.placeholders)

    def get_feed_dict(self):
        """
        Return the feed_dict that is used in super.thread_main() to feed the placeholders.
        :return: The feed_dict.
        """
        return self.get_feed_dict_single()

    def dequeue(self):
        """
        Return the dequeue operation.
        :return: The dequeue operation.
        """
        return self.queue.dequeue_many(self.batch_size)
