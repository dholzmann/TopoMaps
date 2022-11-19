import numpy as np
from typing import Dict, Tuple, List, Iterable, Set


class Edge:
    def __init__(self, nodes: Tuple[int, int], directed=False):
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
    def __init__(self, data_dim: int, nodes: np.array, edges: Dict[int, Set[Edge]]):
        """
        Simple Network class
        :param data_dim: dimensionality of the data
        :param nodes: nodes as np.array. shape = (num_data_points, data_dim)
        :param edges: List of edges
        :param neighbourhood_function: The function to calculate the neighbourhood
        :param learning_rate:
        :param distance_measurement_function:
        """
        if nodes and edges:
            self.nodes = nodes
        else:
            self.nodes = np.empty((0, data_dim))
        if edges:
            self.edges = edges
        else:
            self.edges = {}

    def get_edges(self, node_idx) -> Set[Edge]:
        return self.edges[node_idx]

    def add_node(self, data, neighbours: Iterable[int]=None):
        self.nodes = np.vstack([self.nodes, data])
        node_idx = self.nodes.shape[0]
        if neighbours:
            edges = set([])
            for n in neighbours:
                new_edge = Edge((node_idx, n))
                self.edges[n].add(new_edge)
                edges.add(new_edge)
            self.edges[node_idx] = edges


class SOM:
    def __init__(self, network: Network, learning_rate: float, neighbourhood_function, distance_measurement_function):
        self.network = network
        self.learning_rate = learning_rate
        self.neighbours = neighbourhood_function
        self.dist = distance_measurement_function
