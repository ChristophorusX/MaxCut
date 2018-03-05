from __future__ import print_function
from builtins import range
from future.utils import iteritems
import numpy as np
from propagation_node import PropagationNode


class Graph(object):
    """
    Builds up a graph and operates belief propagation.
    """

    def __init__(self, adjacency, n_clusters, p):
        self.var = {}
        self.dims = []
        self.converged = False
        self.adjacency = adjacency
        self.n_clusters = n_clusters
        self.p = p

    def add_propagation_node(self, n, name, dim):
        """
        Returns a new propagation node to the graph.
        """

        newId = len(self.var)
        new_node = PropagationNode(name, n, dim, newId)
        self.var[name] = new_node
        self.dims.append(dim)

        return new_node

    def construct_from_adj(self):
        """
        Constructs the graph from the given adjacency matrix.
        """

        n, _ = self.adjacency.shape
        for i in range(n):
            n_nbrs = int(np.linalg.norm(self.adjacency[i], ord=1) - 1)  # we assume diagonals being 1
            name = "Node #{}".format(i + 1)
            self.add_propagation_node(n_nbrs, name, self.n_clusters)

        for k, v in self.var.items():
            v.add_nbrs(self)

    def disable_all(self):
        """
        Disables all nodes in graph.
        """

        for k, v in iteritems(self.var):
            v.disable()

    def reset(self):
        """
        Resets messages to original state.
        """

        for k, v in iteritems(self.var):
            v.reset()
        self.converged = False

    def belief_propagation(self, maxsteps=500):
        """
        Belief propagation algorithm.

        On each step:
        - take incoming messages and compute the outgoing accordingly
        - then push outgoing to neighbors' incoming
        - check outgoing with previous outgoing to check for convergence
        """

        # loop to convergence
        timestep = 0
        while timestep < maxsteps and not self.converged:  # run for maxsteps cycles
            timestep += 1
            print("TIME STEP: {}".format(timestep))

            for k, v in iteritems(self.var):
                v.prep_messages(self.p, self.adjacency)

            for k, v in iteritems(self.var):
                v.send_messages()

            # # check for convergence
            # t = True
            # for k, v in iteritems(self.var):
            #     t = t and v.check_convergence()
            #     if not t:
            #         break
            #
            # if t:  # we have convergence!
            #     self.converged = True
            #     print("We have converged!")

        # if run for 500 steps and still no convergence:impor
        if not self.converged:
            print("The belief propagation algorithm does not converge!")

    def marginals(self, maxsteps=500):
        """
        Returns dictionary of all marginal distributions
        indexed by corresponding variable name.
        """

        n, _ = self.adjacency.shape
        # message pass
        self.belief_propagation(maxsteps)

        marginals = np.empty((n, self.n_clusters))
        i = 0
        # for each var
        for k, v in self.var.items():
            if v.enabled:  # only include enabled variables
                aux = v.incoming.dot(self.p)
                # aux[i, :] = np.ones(self.n_clusters)
                # arr = self.adjacency[i, :].reshape((-1, 1))
                # filtering = np.hstack((arr, arr))
                marginals[i] = np.prod(aux, axis=0)
                i += 1

        # normalize
        normalizer = np.sum(marginals, axis=1)
        for r in range(self.n_clusters):
            marginals[:, r] = marginals[:, r] / normalizer

        return marginals
