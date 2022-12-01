
from graph.node import Node


class RunGraph(object):
    """
    Class that runs a graph for given Nodes. For every Node, its parents are evaluated before
    such that every Node in the graph is calculated exactly once.
    Pre-calculated values for a Node in the graph may be given with feed_dict. In this case, neither the given Node,
    nor its parents are being calculated.
    TODO: currently all Node values are cached, which could increase memory consumption. Implement deletion of calculated objects when they are not needed anymore.
    """
    def __init__(self, fetches, feed_dict=None, cache=None, cache_unique_id_node=None):
        """
        Initializer.
        :param fetches: List of Nodes.
        :param feed_dict: Dictionary of pre-calculated Nodes.
        :param cache: The cache object. Must support dictionary interface.
        :param cache_unique_id_node: The node that will be used to obtain the unique_id used for caching.
        """
        self.fetches = fetches
        # initialize node_values with feed_dict
        self.node_values = {}
        if feed_dict is not None:
            self.node_values.update(feed_dict)
        # initialize node_queue with fetches that are not already in the
        # node_values or have already been added to the node_queue
        self.node_queue = []
        for fetch in self.fetches:
            if fetch not in self.node_values and fetch not in self.node_queue:
                self.node_queue.append(fetch)
        self.cache = cache
        self.cache_unique_id_node = cache_unique_id_node
        if self.cache_unique_id_node is not None and self.cache_unique_id_node not in self.node_values and self.cache_unique_id_node not in self.node_queue:
            self.node_queue.append(self.cache_unique_id_node)

    def set_node_value(self, node, value):
        """
        Set the node value.
        :param node: The node.
        :param value: The node value.
        """
        assert node not in self.node_queue, 'The current node has already been calculated. This should not have happened, fix the graph. node = ' + str(node)
        self.node_values[node] = value

    def get_node_value(self, node):
        """
        Get the node value.
        :param node: The node.
        :return: The node value.
        """
        return self.node_values[node]

    def queue_is_empty(self):
        """
        Return True, if the queue is empty.
        :return: True, if the queue is empty.
        """
        return len(self.node_queue) == 0

    def append_to_queue(self, node):
        """
        Append the current node to the queue. If the node is somewhere in node_queue, delete it such that it will not be calculated twice.
        :param node: The node.
        """
        if node in self.node_queue:
            self.node_queue.remove(node)
        # add parent as next node to the queue
        self.node_queue.append(node)

    def pop_from_queue(self):
        """
        Pop a node from the queue.
        :return: The last element of the node_queue.
        """
        node = self.node_queue.pop()
        assert isinstance(node, Node), 'The current node is not a Node object. Either set its value via feed_dict or fix the graph. node = ' + str(node)
        return node

    def get_unique_id(self):
        """
        Return the unique_id of the current evaluated graph instance.
        :return: The unique_id.
        """
        node_value = self.get_node_value(self.cache_unique_id_node)
        assert 'unique_id' in node_value, 'unique_id not in self.cache_unique_id_node values, node_value = ' + str(node_value)
        return node_value['unique_id']

    def node_is_cached(self, node):
        """
        Return True, if the node value is cached and the cache exists.
        :param node: The node for which to check the value.
        :return: True, if the node is cached.
        """
        if self.cache is not None and node.use_caching:
            unique_id = self.get_unique_id()
            name = f'{node.name}_{unique_id}'
            if name in self.cache:
                return True
        return False

    def get_node_from_cache(self, node):
        """
        Return the node value from the cache, if it is cached and the cache exists.
        :param node: The node for which to return the value.
        :return: The node value or None.
        """
        if self.cache is not None and node.use_caching:
            unique_id = self.get_unique_id()
            name = f'{node.name}_{unique_id}'
            if name in self.cache:
                value = self.cache[name]
                return value
        return None

    def add_node_to_cache(self, node, value):
        """
        Add a node to the cache, if the node should be cached and a cache exists.
        :param node: The node to cache.
        :param value: The value to cache.
        """
        if self.cache is not None and node.use_caching:
            unique_id = self.get_unique_id()
            name = f'{node.name}_{unique_id}'
            self.cache[name] = value

    def check_all_parents_calculated(self, node):
        """
        Check if parents are already calculated.
        :param node: The node for which to check the parents.
        :return: True, if all parent values have been calculated.
        """
        for parent in list(node.get_parents()) + list(node.get_kwparents().values()):
            if parent not in self.node_values:
                return False
        return True

    def append_missing_parents_to_queue(self, node):
        """
        Append not calculated parents of node to node_queue.
        :param node: The node for which to check the parents.
        """
        for parent in list(node.get_parents()) + list(node.get_kwparents().values()):
            if parent not in self.node_values:
                self.append_to_queue(parent)

    def get_parent_values(self, node):
        """
        Return parents_values, kwparents_values tuple of the values of the parents.
        :param node: The node for which to return the parents.
        :return: parents_values, kwparents_values tuple.
        """
        all_parent_nodes = list(node.get_parents()) + list(node.get_kwparents().values())
        if len(all_parent_nodes) == 1 and isinstance(self.node_values[all_parent_nodes[0]], Node):
            # replace current parent node with parents of all_parent_node
            # this is very hacky, but allows to incorporate conditions into the graph
            only_parent_node = self.node_values[all_parent_nodes[0]]
            parents_values = [self.node_values[parent] for parent in only_parent_node.get_parents()]
            kwparents_values = dict([(parent_key, self.node_values[parent]) for parent_key, parent in only_parent_node.get_kwparents().items()])
        else:
            # set parents to fetched objects
            parents_values = [self.node_values[parent] for parent in node.get_parents()]
            kwparents_values = dict([(parent_key, self.node_values[parent]) for parent_key, parent in node.get_kwparents().items()])
        return parents_values, kwparents_values

    def calculate_node(self, node):
        """
        Execute a node and return its value. The parents must have been already calculated!
        :param node: The node to calculate.
        :return: The calculated value.
        """
        parents_values, kwparents_values = self.get_parent_values(node)
        return node.get(*parents_values, **kwparents_values)

    def run(self):
        """
        Function that runs a graph for given Nodes. For every Node, its parents are evaluated before
        such that every Node in the graph is calculated exactly once.
        :return: The calculated values in the same order as fetches.
        """
        while not self.queue_is_empty():
            current_node = self.pop_from_queue()

            # check if Node is in cache
            if self.node_is_cached(current_node):
                # if so, just use the cached value and continue with next node
                current_node_value = self.get_node_from_cache(current_node)
                self.set_node_value(current_node, current_node_value)
                continue

            # check if all parents have been calculated
            if not self.check_all_parents_calculated(current_node):
                self.append_to_queue(current_node)
                self.append_missing_parents_to_queue(current_node)
                continue

            # calculate current node
            current_node_value = self.calculate_node(current_node)
            self.set_node_value(current_node, current_node_value)
            self.add_node_to_cache(current_node, current_node_value)

            # if current_output is a Node object, put it into the node_queue as it probably needs to be processed
            if isinstance(current_node_value, Node):
                self.append_to_queue(current_node_value)

        return [self.get_node_value(fetch) for fetch in self.fetches]
