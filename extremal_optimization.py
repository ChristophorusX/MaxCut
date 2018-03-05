import numpy as np
import re
import heapq
from random import randint
import scipy.stats as stats
import propagation_node as ppn
import propagation_graph as ppg


class EONode(ppn.Node):
    """
    EONode is a node class for extremal optimization algorithm.
    """

    def __init__(self, name, n, dim, nid):
        super(EONode, self).__init__(nid)
        self.name = name
        self.dim = dim
        self.incoming = None
        self.outgoing = None
        self.oldoutgoing = None
        self.observed = -1  # only >= 0 if variable is observed
        self.fitness = 0
        self.grouping = None

    def __lt__(self, other):
        return self.fitness < other.fitness

    def reset(self):
        super(EONode, self).reset()
        # n, q = self.incoming.shape
        self.incoming = None
        self.outgoing = None
        self.oldoutgoing = None
        self.observed = -1
        self.fitness = 0
        self.grouping = None

    def add_nbrs(self, graph):
        """
        Adds all the neighbors of an eonode given the information of the graph.
        """

        pattern = re.compile(r"[0-9]+")
        index = int(re.findall(pattern, self.name)[0]) - 1
        n, _ = graph.adjacency.shape
        nbr_arr = graph.adjacency[index]
        for i in range(n):
            if i != index and nbr_arr[i] != 0:
                self.nbrs.append(graph.var["Node #{}".format(i + 1)])

    def update_fitness(self):
        """
        Updates the fitness value of a node given the groupings of its neighbors.
        """

        if not self.nbrs:
            self.fitness = 1
        else:
            good = 0
            bad = 0
            for node in self.nbrs:
                if self.grouping == node.grouping:
                    good += 1
                else:
                    bad += 1
            self.fitness = good / (good + bad)


class EOGraph(ppg.Graph):
    """
    EOGraph is a graph containing EONodes.
    """

    def __init__(self, adjacency, n_clusters, p):
        super(EOGraph, self).__init__(adjacency, n_clusters, p)
        self.heap = None

    def add_eo_node(self, n, name, dim):
        newId = len(self.var)
        new_node = EONode(name, n, dim, newId)
        self.var[name] = new_node
        self.dims.append(dim)

        return new_node

    def construct_from_adj(self):
        """
        Constructs the eograph from the given adjacency matrix.
        """

        n, _ = self.adjacency.shape
        for i in range(n):
            # we assume diagonals being 1
            n_nbrs = int(np.linalg.norm(self.adjacency[i], ord=1) - 1)
            name = "Node #{}".format(i + 1)
            self.add_eo_node(n_nbrs, name, self.n_clusters)

        for k, v in self.var.items():
            v.add_nbrs(self)

    def initialize_grouping(self):
        """
        Randomly assigns the grouping to each node.
        """

        n, _ = self.adjacency.shape
        grouping1 = 0
        grouping2 = 0
        for key, node in self.var.items():
            if grouping1 < int(n / 2) and grouping2 < int(n / 2):
                node.grouping = randint(1, 2)
                if node.grouping == 1:
                    grouping1 += 1
                else:
                    grouping2 += 1
            elif grouping1 >= int(n / 2):
                node.grouping = 2
                grouping2 += 1
            else:
                node.grouping = 1
                grouping1 += 1

    def sorting_by_fitness(self):
        """
        Sorts the nodes in a min heap by their fitness.
        """

        if self.heap is None:
            self.build_heap_from_dict()
        else:
            self.heap.sort()

    def build_heap_from_dict(self):
        """
        Builds a heap from the dictionary that contains all the vertices.
        """

        self.heap = [node for key, node in self.var.items()]
        heapq.heapify(self.heap)  # min heap

    def random_swapping(self, tau):
        """
        Randomly swaps two vertices in different groups with a power law chance.
        """

        n, _ = self.adjacency.shape
        distribution = self.truncated_power_law(tau, n)
        k1 = distribution.rvs()
        node1 = self.heap[k1 - 1]
        grouping1 = node1.grouping
        k2 = distribution.rvs()
        while self.heap[k2 - 1].grouping == grouping1:
            k2 = distribution.rvs()
        node2 = self.heap[k2 - 1]
        tmp = node1.grouping
        node1.grouping = node2.grouping
        node2.grouping = tmp
        return node1, node2

    def truncated_power_law(self, tau, m):
        """
        Generates the distribution according to a power law controlled by
        parameter tau.
        """

        x = np.arange(1, m + 1, dtype='float')
        pmf = 1 / x**tau
        pmf /= pmf.sum()
        return stats.rv_discrete(values=(range(1, m + 1), pmf))

    def extremal_optimization(self, tau):
        """
        The Extremal Optimization algorithm.
        """

        self.initialize_grouping()
        self.sorting_by_fitness()
        n, _ = self.adjacency.shape
        t_max = 20 * n  # TODO: customize const 1 << A << n
        for i in range(t_max):
            print("ROUND {}".format(i))
            node1, node2 = self.random_swapping(tau)
            node1.update_fitness()
            node2.update_fitness()
            for node in node1.nbrs:
                node.update_fitness()
            for node in node2.nbrs:
                node.update_fitness()
            self.sorting_by_fitness()
        result = [node.grouping for key, node in self.var.items()]
        return result
