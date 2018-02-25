from __future__ import print_function
from builtins import range
from future.utils import iteritems
import numpy as np
from node import PropagationNode


class Graph(object):
    """
    Builds up a graph and operates belief propagation.
    """

    def __init__(self, adjacency):
        self.var = {}
        self.dims = []
        self.converged = False
        self.adjacency = adjacency

    def add_propagation_node(self, name, dim):
        newId = len(self.var)
        new_node = PropagationNode(name, dim, newId)
        self.var[name] = new_node
        self.dims.append(dim)

        return new_node

    def disable_all(self):
        """
        Disable all nodes in graph.
        """
        for k, v in iteritems(self.var):
            v.disable()

    def reset(self):
        """
        Reset messages to original state.
        """
        for k, v in iteritems(self.var):
            v.reset()
        # for f in self.fac:
        #     f.reset()
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
                v.prep_messages()

            for k, v in iteritems(self.var):
                v.send_messages()

            # check for convergence
            t = True
            for k, v in iteritems(self.var):
                t = t and v.check_convergence()
                if not t:
                    break

            if t:  # we have convergence!
                self.converged = True

        # if run for 500 steps and still no convergence:impor
        if not self.converged:
            print("The belief propagation algorithm does not converge!")

    def marginals(self, adjacency, n, q, p, maxsteps=500):
        """
        Returns dictionary of all marginal distributions
        indexed by corresponding variable name.
        """
        # message pass
        self.belief_propagation(maxsteps)

        marginals = np.empty((n, q))
        i = 0
        # for each var
        for k, v in iteritems(self.var):
            if v.enabled:  # only include enabled variables
                aux = v.incoming.dot(p)
                aux[i, :] = np.ones(q)
                arr = adjacency[i, :].reshape((-1, 1))
                filtering = np.hstack((arr, arr))
                marginals[i] = np.prod(aux * filtering, axis=0)
                i += 1

        # normalize
        normalizer = np.sum(marginals, axis=1).reshape((-1, 1))
        for r in range(q):
            marginals[:, r] = marginals[:, r] / normalizer

        return marginals
