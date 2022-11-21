import numpy as np
from abc import ABC, abstractmethod
from typing import DefaultDict, Tuple, Iterable, Set
from network.neighbourhood_functions import Gaussian
from math import floor
from collections import defaultdict


def default_dict_val():
    return set()


class Edge:
    def __init__(self, nodes: Tuple[int, int], directed=False):
        assert nodes
        self.age = 0
        self.nodes = nodes
        self.directed = directed

    def __repr__(self):
        if self.directed:
            return f'({self.nodes[0]} --> {self.nodes[1]})'
        return f'({self.nodes[0]} <--> {self.nodes[1]})'

    def __hash__(self):
        if self.directed:
            return hash(self.nodes)
        return hash(tuple(sorted(self.nodes)))

    def __eq__(self, other):
        if self.directed:
            return self.nodes == other
        return self.nodes == other or self.nodes == other[::-1]

    def get_other(self, node: int):
        assert node in self.nodes
        if node == self.nodes[0]:
            return self.nodes[1]
        return self.nodes[0]


class Network:
    def __init__(self, data_dim: int, nodes: np.array, edges: DefaultDict[int, Set[Edge]]):
        """
        Simple Network class
        :param data_dim: dimensionality of the data
        :param nodes: nodes as np.array. shape = (num_data_points, data_dim)
        :param edges: List of edges
        """
        if len(nodes) != 0:
            self.nodes = nodes
        else:
            self.nodes = np.empty((0, data_dim))
        if len(edges) != 0:
            self.edges = edges
        else:
            self.edges = defaultdict(default_dict_val)

    def get_edges(self, node_idx) -> Set[Edge]:
        return self.edges[node_idx]

    def get_nodes(self) -> np.array:
        return self.nodes

    def add_node(self, data, neighbours: Iterable[int] = None):
        self.nodes = np.vstack([self.nodes, data])
        node_idx = self.nodes.shape[0]
        if neighbours:
            edges = set([])
            for n in neighbours:
                new_edge = Edge((node_idx, n))
                self.edges[n].add(new_edge)
                edges.add(new_edge)
            self.edges[node_idx] = edges

    def add_edge(self, node1, node2, directed=False):
        new_edge = Edge((node1, node2))
        self.edges[node1].add(new_edge)
        if not directed:
            self.edges[node2].add(new_edge)

    def remove_node(self, idx):
        np.delete(self.nodes, idx, axis=0)


class LearnableNetwork(ABC):
    """
    Abstract class for learnable networks / topological maps.
    Contains functions that have to be implemented and functions that may be optional
    """
    @abstractmethod
    def weight_update(self, n: int, x: np.array):
        """abstract method => need to be overwritten"""
        pass

    @abstractmethod
    def get_winner(self, x: np.array) -> int:
        """abstract method => need to be overwritten"""
        pass

    @abstractmethod
    def process_input(self, data_points: np.array, random_shuffle=False):
        """abstract method => need to be overwritten"""
        pass

    def edge_update(self):
        """ optional - does not need to be overwritten if not needed"""
        pass


class SOM(LearnableNetwork):
    def __init__(self,
                 data_dim,
                 num_row=10,
                 num_col=10,
                 random_interval: Tuple[float, float] = (0, 1),
                 learning_rate_initial: float = 0.1,
                 learning_rate_final: float = 0.005,
                 lr_step=10000,
                 neighbourhood_function=Gaussian()):
        self.rows = num_row
        self.columns = num_col
        self.network = SOM.init_2d_grid_network(num_row, num_col, data_dim, random_interval)
        self.learning_rate = learning_rate_initial
        self.learning_rate_final = learning_rate_final
        self.lr_step = (learning_rate_final-self.learning_rate) / lr_step
        self.neighbour_func = neighbourhood_function
        self.dist = lambda x: np.linalg.norm(x)

    @staticmethod
    def init_2d_grid_network(num_row, num_col, data_dim, rand_interval: Tuple[float, float] = (0, 1)) -> Network:
        nodes = (rand_interval[1] - rand_interval[0]) * np.random.random_sample((num_row*num_col, data_dim)) + rand_interval[0]
        network = Network(data_dim, nodes, {})
        # create grid structure
        for x in range(num_row):
            for y in range(num_col):
                idx = x*num_col + y
                if x > 0:
                    network.add_edge(idx, (x-1)*num_col + y)
                if y > 0:
                    network.add_edge(idx, x * num_col + y-1)
        return network

    def process_input(self, data_points: np.array, random_shuffle=False):
        if random_shuffle:
            np.random.shuffle(data_points)
        for x in data_points:
            n = self.get_winner(x)
            self.weight_update(n, x)
            self.decrease_learning_rate()

    def get_winner(self, x) -> int:
        return np.argmin(np.array(list(map(self.dist, self.network.nodes-x))))

    def get_idx(self, n: int) -> np.array:
        if n == 0:
            return np.array([0, 0])
        n_x = floor(n/self.columns)
        n_y = n - n_x * self.rows
        return np.array([n_x, n_y])

    def weight_update(self, n, x):
        for c, node in enumerate(self.network.nodes):
            update = self.learning_rate * self.neighbour_func(self.get_idx(n), self.get_idx(c)) * (x - node)
            self.network.nodes[c] += update
            #print(self.learning_rate)

    def decrease_learning_rate(self):
        if self.learning_rate + self.lr_step <= self.learning_rate_final:
            self.learning_rate = self.learning_rate_final
        else:
            self.learning_rate += self.lr_step


