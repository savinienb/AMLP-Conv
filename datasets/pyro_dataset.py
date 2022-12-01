
import datetime

import Pyro4
import numpy as np

import utils.compression
from datasets.dataset_base import DatasetBase
from datasets.dataset_queue import DatasetQueue

Pyro4.config.COMPRESSION = False
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED = {'pickle'}
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


@Pyro4.expose
class PyroServerDataset(DatasetBase):
    """
    Pyro server dataset that prefetches entry dicts for an encapsulated dataset (self.dataset).
    Needs to be derived in order to set the dataset and to set it as a Pyro object.
    """
    def __init__(self, queue_size=128, refill_queue_factor=0.0, n_threads=8, use_multiprocessing=False, compression_type=None):
        """
        Initializer.
        :param queue_size: The number of entries in the queue used for caching.
        :param refill_queue_factor: If the number of entries in the queue is less than queue_size*refill_queue_factor, the entry
                                    that is returned with get() will be again put into the end of the queue.
        :param n_threads: The number of prefetching threads.
        :param use_multiprocessing: If true, use processes instead of threads. May be faster, as it circumvents the GIL, but may also use much more memory, as less memory is shared.
        :param compression_type: The used compression type. See utils.compression.
        """
        self.queue_size = queue_size
        self.refill_queue_factor = refill_queue_factor
        self.n_threads = n_threads
        self.use_multiprocessing = use_multiprocessing
        self.compression_type = compression_type
        self.dataset_queue = None
        self.dataset = None
        self.args = []
        self.kwargs = {}

    def __del__(self):
        self.close_queue()

    def initialized_with_same_parameters(self, args, kwargs):
        """
        Return true, if the previous call of the PyroDataset was initialized with the same dataset args and kwargs.
        :param args: args that were used for initializing the internal dataset.
        :param kwargs: kwargs that were used for initializing the internal dataset.
        :return: True, if same args and kwargs, False otherwise.
        """
        return list(self.args) == list(args) and dict(self.kwargs) == dict(kwargs)

    def init_with_parameters(self, *args, **kwargs):
        """
        Method that gets called, after getting the Proxy object from the server.
        Overwrite, if needed.
        """
        pass

    def set_compression_type(self, compression_type):
        """
        Set the compression type.
        :param compression_type: The used compression type. None, 'lz4', or 'zfpy'.
        """
        self.compression_type = compression_type

    def init_queue(self):
        """
        Starts the prefetching threads.
        """
        self.dataset_queue = DatasetQueue(self.dataset,
                                          queue_size=self.queue_size,
                                          n_threads=self.n_threads,
                                          use_multiprocessing=self.use_multiprocessing,
                                          queue_compression_type=self.compression_type,
                                          cache_compression_type=self.compression_type,
                                          use_shared_memory_queue=False,
                                          use_shared_memory_cache=True,
                                          use_caching=True,
                                          print_queue_size=True)

    def close_queue(self):
        """
        Stops and joins the prefetching threads.
        """
        if self.dataset_queue is None:
            return
        self.dataset_queue.close()
        self.dataset_queue = None

    def get_next(self, *args, **kwargs):
        """
        Returns the next entry dict. This function is called by PyroClientDataset().get_next()
        :return: The next entry dict of the internal queue.
        """
        error, done, queue_entry = self.dataset_queue.get_queue_entry()
        if error:
            raise error
        return queue_entry

    def num_entries(self):
        """
        Not supported.
        """
        raise RuntimeError('num_entries() is not supported for PyroServerDataset.')

    def get(self, id_dict, *args, **kwargs):
        """
        Not supported.
        """
        raise RuntimeError('get(id_dict) is not supported for PyroServerDataset. Use get_next() instead.')


class PyroClientDataset(DatasetBase):
    """
    Pyro client dataset that encapsulate a Pyro server dataset at the given uri.
    """
    def __init__(self, uri, compression_type=None, *args, **kwargs):
        """
        Gets the server dataset at the given URI and stops and starts its threads.
        :param uri: URI to connect to.
        :param compression_type: The used compression type. None, 'lz4', or 'zfpy'.
        :param args: Arguments passed to init_with_parameters.
        :param kwargs: Keyword arguments passed to init_with_parameters.
        """
        self.uri = uri
        self.compression_type = compression_type
        self.server_dataset = Pyro4.Proxy(self.uri)
        #if not self.server_dataset.initialized_with_same_parameters(args, kwargs):
        self.server_dataset.close_queue()
        self.server_dataset.init_with_parameters(*args, **kwargs)
        #self.server_dataset.set_compression_type(self.compression_type)
        self.server_dataset.init_queue()

    def queue_entry_to_dataset_entry(self, queue_entry):
        """
        Convert the queue entries that were transmitted through pyro to the dataset entries.
        :param queue_entry: The queue entry.
        :return: A dict that contains the converted entries as dataset_entry['generators'].
        """
        dataset_entry = utils.compression.decompress_and_unpickle(queue_entry, compression=self.compression_type)
        return dataset_entry

    def get_next(self, *args, **kwargs):
        """
        Returns the next entry dict of the server_dataset.
        :return: The next entry dict of the server_dataset.
        """
        queue_entry = self.server_dataset.get_next()
        return self.queue_entry_to_dataset_entry(queue_entry)

    def num_entries(self):
        """
        Not supported.
        """
        raise RuntimeError('num_entries() is not supported for PyroClientDataset.')

    def get(self, id_dict, *args, **kwargs):
        """
        Not supported.
        """
        raise RuntimeError('get(id_dict) is not supported for PyroClientDataset. Use get_next() instead.')
