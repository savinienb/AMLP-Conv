import datetime
import multiprocessing
import threading
import traceback
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager, SyncManager
import queue

import numpy as np
import utils.compression

from collections.abc import MutableMapping

import random


class SharedMemoryCache(MutableMapping):
    """
    Multiprocessing cache that uses shared memory and pickle compression. Already set items are *not* overwritten.
    """
    def __init__(self, shared_lock, shared_dict, shared_memory_manager, compression=None):
        self.lock = shared_lock
        self.store = shared_dict
        self.shared_memory_manager = shared_memory_manager
        self.compression = compression

    def __del__(self):
        with self.lock:
            for key in self.store.keys():
                try:
                    shm = shared_memory.SharedMemory(name=self.store[key], create=False)
                    shm.unlink()
                except Exception:
                    pass

    def __getitem__(self, key):
        with self.lock:
            shm = shared_memory.SharedMemory(name=self.store[key], create=False)
        unpickled_value = utils.compression.decompress_and_unpickle(shm.buf, self.compression)
        return unpickled_value

    def __setitem__(self, key, value):
        pickled_value = utils.compression.pickle_and_compress(value, self.compression)
        with self.lock:
            if key in self.store:
                return
            shm = self.shared_memory_manager.SharedMemory(len(pickled_value))
            shm.buf[:] = pickled_value
            self.store[key] = shm.name

    def __delitem__(self, key):
        with self.lock:
            if key in self.store:
                shm = shared_memory.SharedMemory(name=self.store[key], create=False)
                shm.unlink()
            del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class CompressionCache(MutableMapping):
    """
    Multiprocessing cache that uses pickle compression. Already set items are *not* overwritten.
    """
    def __init__(self, shared_lock, shared_dict, compression=None):
        self.lock = shared_lock
        self.store = shared_dict
        self.compression = compression

    def __getitem__(self, key):
        with self.lock:
            pickled_value = self.store[key]
        unpickled_value = utils.compression.decompress_and_unpickle(pickled_value, self.compression)
        return unpickled_value

    def __setitem__(self, key, value):
        pickled_value = utils.compression.pickle_and_compress(value, self.compression)
        with self.lock:
            if key in self.store:
                return
            self.store[key] = pickled_value

    def __delitem__(self, key):
        with self.lock:
            del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class DatasetQueueWorker(object):
    def __init__(self,
                 dataset,
                 n_threads=None,
                 use_multiprocessing=True,
                 dataset_entries_to_keep=None,
                 queue_compression_type=None,
                 use_shared_memory_queue=False,
                 shared_memory_manager=None,
                 print_queue_size=False,
                 print_process_state=True,
                 repeat=True):
        self.dataset = dataset
        self.n_threads = n_threads
        self.use_multiprocessing = use_multiprocessing
        self.dataset_entries_to_keep = dataset_entries_to_keep or ['generators']
        self.queue_compression_type = queue_compression_type
        self.use_shared_memory_queue = use_shared_memory_queue
        self.shared_memory_manager = shared_memory_manager
        self.print_queue_size = print_queue_size
        self.print_process_state = print_process_state
        self.repeat = repeat

    def dataset_entry_to_queue_entry(self, dataset_entry):
        """
        Convert a dataset entry to a queue entry. The dataset entry will be filtered, pickled and compressed.
        :param dataset_entry: The dataset entry.
        :return: The pickled queue entry.
        """
        filtered_dataset_entry = {}
        if self.dataset_entries_to_keep is not None:
            for entry_to_keep in self.dataset_entries_to_keep:
                filtered_dataset_entry[entry_to_keep] = dataset_entry[entry_to_keep]
        else:
            filtered_dataset_entry = dataset_entry
        if self.use_shared_memory_queue:
            pickle_dump = utils.compression.pickle_and_compress(filtered_dataset_entry, compression=self.queue_compression_type)
            shm = self.shared_memory_manager.SharedMemory(len(pickle_dump))
            shm.buf[:] = pickle_dump
            shm.close()
            return shm.name
        elif self.queue_compression_type is not None:
            return utils.compression.pickle_and_compress(filtered_dataset_entry, compression=self.queue_compression_type)
        else:
            return filtered_dataset_entry

    def run(self, process_id, queue, should_stop, cache):
        """
        Main function of the prefetching processes/threads.
        """
        np.random.seed()
        if not self.repeat:
            indizes = list(range(process_id, self.dataset.num_entries(), self.n_threads))
            curr_index_index = 0
        else:
            indizes = range(process_id % self.dataset.num_entries(), self.dataset.num_entries(), self.n_threads)
        if self.print_process_state:
            type_string = 'process' if self.use_multiprocessing else 'thread'
            print(f'DatasetQueue begin loop {type_string} {process_id}')
        try:
            while not should_stop.value:
                try:
                    if not self.repeat:
                        if curr_index_index >= len(indizes):
                            break
                        curr_index = indizes[curr_index_index]
                        curr_index_index += 1
                        dataset_entry = self.dataset.get_for_index(curr_index, cache=cache)
                    else:
                        # dataset_entry = self.dataset.get_next(cache=cache)
                        image_index = random.choice(indizes)
                        dataset_entry = self.dataset.get_for_index(image_index, cache=cache)
                    queue_entry = self.dataset_entry_to_queue_entry(dataset_entry)
                except Exception as e:
                    # print all exceptions and put them into queue such that the main process
                    # could handle them
                    traceback.print_exc()
                    queue.put((e, False, None))
                    should_stop.value = True
                    break
                if should_stop.value:
                    break
                if self.print_queue_size:
                    print(f'put {queue.qsize()} (pid {process_id}) {datetime.datetime.now()}')
                queue.put((None, False, queue_entry))
        except BaseException:
            # in case of BaseException, exit loop and stop process
            # if the queue is empty, and the current process is the first one, add an entry to wake waiting threads
            if queue.empty() and process_id == 0:
                queue.put_nowait((None, True, None))
            pass

        if self.print_process_state:
            print(f'DatasetQueue end loop {type_string} {process_id}')


class DatasetQueue(object):
    """
    Dataset queue that performs prefetching in either background threads or processes.
    """
    def __init__(self,
                 dataset,
                 queue_size=32,
                 n_threads=None,
                 use_multiprocessing=True,
                 dataset_entries_to_keep=None,
                 queue_compression_type=None,
                 cache_compression_type='lz4',
                 use_caching=True,
                 use_shared_memory_queue=False,
                 use_shared_memory_cache=True,
                 print_queue_size=False,
                 print_process_state=True,
                 repeat=True):
        """
        Initializer.
        :param dataset: The dataset used for data generation.
        :param queue_size: The queue_size.
        :param n_threads: The number of background threads. If None, then calculate n_threads based on CPU cores.
        :param use_multiprocessing: If True, use multiprocessing for workers, otherwise use threads.
                                    Multiprocessing should be much faster, but could increase memory consumption.
        :param dataset_entries_to_keep: List of dataset entry keys to keep. If None, only 'generators' will be kept.
        :param queue_compression_type: Compression type used for entries in the queue. See utils.compression.
        :param cache_compression_type: Compression type used for entries in the cache. See utils.compression.
        :param use_caching: If True, use a cache when calling self.dataset.get_next().
        :param use_shared_memory_queue: If True, use shared memory for putting/getting queue entries.
        :param use_shared_memory_queue: If True, use shared memory for cache.
        :param print_queue_size: If True, print every queue get/set.
        :param print_process_state: If True, print process state output, i.e., process start/stop/etc.
        :param repeat: If True, return dataset entries indefinitely. Otherwise, just iterate once over the given
                       dataset - useful for validation dataset. If False, 'dataset' must implement the methods
                       num_entries() and get_for_index().
        """
        self.workers_running = False
        self.dataset = dataset
        self.queue_size = queue_size
        self.n_threads = n_threads
        self.use_multiprocessing = use_multiprocessing
        self.dataset_entries_to_keep = dataset_entries_to_keep or ['generators']
        self.queue_compression_type = queue_compression_type
        self.cache_compression_type = cache_compression_type
        self.use_caching = use_caching
        self.print_queue_size = print_queue_size
        self.print_process_state = print_process_state
        self.repeat = repeat
        # variables for managing background processes/threads
        self.multiprocessing_manager = None
        if self.use_multiprocessing:
            self.multiprocessing_manager = SyncManager()
            self.multiprocessing_manager.start()
            self.should_stop = self.multiprocessing_manager.Value('b', False)
            self.queue = self.multiprocessing_manager.Queue(maxsize=self.queue_size)
        else:
            class ShouldStop(object):
                def __init__(self):
                    self.value = False
            self.should_stop = ShouldStop()
            self.queue = queue.Queue(maxsize=self.queue_size)
        self.use_shared_memory_queue = use_shared_memory_queue
        self.use_shared_memory_cache = use_shared_memory_cache
        self.shared_memory_manager = None
        if self.use_shared_memory_queue or self.use_shared_memory_cache:
            self.shared_memory_manager = SharedMemoryManager()
            self.shared_memory_manager.start()
        self.cache = None
        self.cache_lock = None
        self.cache_dict = None
        if self.use_caching:
            if self.use_multiprocessing:
                self.cache_lock = self.multiprocessing_manager.Lock()
                self.cache_dict = self.multiprocessing_manager.dict()
            else:
                self.cache_lock = threading.Lock()
                self.cache_dict = {}
            if self.use_shared_memory_cache:
                self.cache = SharedMemoryCache(self.cache_lock, self.cache_dict, self.shared_memory_manager, self.cache_compression_type)
            else:
                self.cache = CompressionCache(self.cache_lock, self.cache_dict, self.cache_compression_type)
        else:
            self.cache = None
        self.threads = []
        self.start_workers()

    def __del__(self):
        """
        Stop workers on deletion.
        """
        self.close()

    def get(self):
        """
        Return the next dataset entry.
        :return: (error, done, dataset_entry) tuple.
        """
        if self.print_queue_size:
            print('get', self.queue.qsize(), datetime.datetime.now())
        error, done, queue_entry = self.queue.get()
        if error or done:
            return error, done, None
        dataset_entry = self.queue_entry_to_dataset_entry(queue_entry)
        return error, done, dataset_entry

    def get_queue_entry(self):
        """
        Return the next raw queue entry. The raw queue entry needs to be pickled and compressed.
        :return: (error, done, queue_entry) tuple.
        """
        if self.print_queue_size:
            print('get', self.queue.qsize(), datetime.datetime.now())
        return self.queue.get()

    def close(self):
        """
        Stop the the data generation loop and clear the queue.
        """
        self.stop_workers()

    def qsize(self):
        """
        Return the current queue size.
        :return: The current queue size.
        """
        return self.queue.qsize()

    def queue_entry_to_dataset_entry(self, queue_entry):
        """
        Convert a queue entry to a dataset entry. The queue entry will be decompressed and unpickled.
        :param queue_entry: The queue entry.
        :return: The dataset entry.
        """
        if self.use_shared_memory_queue:
            shm = shared_memory.SharedMemory(name=queue_entry, create=False)
            value = utils.compression.decompress_and_unpickle(shm.buf, compression=self.queue_compression_type)
            shm.unlink()
            return value
        elif self.queue_compression_type is not None:
            return utils.compression.decompress_and_unpickle(queue_entry, compression=self.queue_compression_type)
        else:
            return queue_entry

    def start_workers(self):
        """
        Start the prefetching threads.
        """
        if self.print_process_state:
            print('DatasetQueue starting...')

        self.should_stop.value = False

        for i in range(self.n_threads):
            worker = DatasetQueueWorker(dataset=self.dataset,
                                        n_threads=self.n_threads,
                                        use_multiprocessing=self.use_multiprocessing,
                                        dataset_entries_to_keep=self.dataset_entries_to_keep,
                                        queue_compression_type=self.queue_compression_type,
                                        use_shared_memory_queue=self.use_shared_memory_queue,
                                        shared_memory_manager=self.shared_memory_manager,
                                        print_queue_size=self.print_queue_size,
                                        print_process_state=self.print_process_state,
                                        repeat=self.repeat)
            if self.use_multiprocessing:
                thread = multiprocessing.Process(target=worker.run, args=(i, self.queue, self.should_stop, self.cache))
            else:
                thread = threading.Thread(target=worker.run, args=(i, self.queue, self.should_stop, self.cache))
            thread.daemon = True  # Process/Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        if self.print_process_state:
            print('DatasetQueue started')

        self.workers_running = True

    def stop_workers(self):
        """
        Stop and joins the prefetching threads.
        """
        if not self.workers_running:
            return

        if self.print_process_state:
            print('DatasetQueue stopping...')

        self.should_stop.value = True
        # at first, empty queue
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass
        # then, join threads
        for thread in self.threads:
            thread.join()
        # then, empty queue again, as threads could have put additional entries
        # NOTE: try ... except should not be needed, as no put should be called after joining
        while not self.queue.empty():
            self.queue.get_nowait()

        # queue is empty and all threads are finished -> fill queue with "finished" entries to signal waiting threads
        # should not be needed, but could help in cases when get is called also after the first "finished" entry was
        # already returned, i.e., multiple threads are waiting.
        for _ in range(self.queue_size):
            self.queue.put_nowait((None, True, None))

        # final clean up
        self.threads = []
        self.queue = None
        self.cache = None
        if self.shared_memory_manager:
            self.shared_memory_manager.shutdown()
            self.shared_memory_manager = None
        if self.multiprocessing_manager:
            self.multiprocessing_manager.shutdown()
            self.multiprocessing_manager = None

        if self.print_process_state:
            print('DatasetQueue stopped')

        self.workers_running = False
