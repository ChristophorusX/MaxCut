import numpy as np
import re
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
        self.incoming = np.random.uniform(0, 1, (n, self.dim))
        self.outgoing = np.random.uniform(0, 1, (n, self.dim))
        self.oldoutgoing = np.random.uniform(0, 1, (n, self.dim))
        self.observed = -1  # only >= 0 if variable is observed
        self.fitness = 0
        self.grouping = None

    def reset(self):
        super(EONode, self).reset()
        n, q = self.incoming.shape
        self.incoming = np.ones((n, self.dim))
        self.outgoing = np.ones((n, self.dim))
        self.oldoutgoing = np.ones((n, self.dim))
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
            n_nbrs = int(np.linalg.norm(self.adjacency[i], ord=1) - 1) # we assume diagonals being 1
            name = "Node #{}".format(i + 1)
            self.add_eo_node(n_nbrs, name, self.n_clusters)

        for k, v in self.var.items():
            v.add_nbrs(self)

    def initialize_grouping(self):
        pass
