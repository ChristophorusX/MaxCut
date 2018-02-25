from builtins import range
import numpy as np


class Node(object):
    """
    Superclass for propagation nodes.
    """

    epsilon = 10**(-4)

    def __init__(self, nid):
        self.enabled = True
        self.nid = nid
        self.nbrs = []
        self.incoming = []
        self.outgoing = []
        self.oldoutgoing = []

    def reset(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True
        for n in self.nbrs:
            # it will recursively enable entire graph
            n.enabled = True

    def next_step(self):
        """
        Copy outgoing massages into oldoutgoing.
        """
        self.oldoutgoing = self.outgoing[:, :]

    def normalize_messages(self):
        """
        Normalize to sum to 1.
        """
        # self.outgoing = [x / np.sum(x) for x in self.outgoing]
        n, q = self.outgoing.shape
        normalizer = np.sum(self.outgoing, axis=1).reshape((-1, 1))
        for r in range(q):
            self.outgoing[:, r] = self.outgoing[:, r] / normalizer

    def receive_message(self, f, m):
        """
        Places new message into correct location in new message list.
        """
        if self.enabled:
            i = self.nbrs.index(f)
            self.incoming[i] = m

    def send_messages(self):
        """
        Sends all outgoing messages.
        """
        n, q = self.outgoing.shape
        for i in range(n):
            self.nbrs[i].receive_message(self, self.outgoing[i])

    def check_convergence(self):
        """
        Check if any messages have changed.
        """
        n, q = self.outgoing.shape
        if self.enabled:
            for i in range(n):
                # check messages have same shape
                self.oldoutgoing[i].shape = self.outgoing[i].shape
                delta = np.linalg.norm(self.outgoing[i] - self.oldoutgoing[i])
                if (delta > Node.epsilon).any():  # if there has been change
                    return False
            return True
        else:
            # Always return True if disabled to avoid interrupting check
            return True


class PropagationNode(Node):
    """
    Node in graph under propagation.
    """

    def __init__(self, name, dim, nid):
        super(PropagationNode, self).__init__(nid)
        self.name = name
        self.dim = dim
        self.observed = -1  # only >= 0 if variable is observed

    def reset(self):
        super(PropagationNode, self).reset()
        n, q = self.incoming.shape
        self.incoming = np.ones((n, self.dim))
        self.outgoing = np.ones((n, self.dim))
        self.oldoutgoing = np.ones((n, self.dim))
        self.observed = -1

    def condition(self, observation):
        """
        Condition on observing certain value.
        """
        self.enable()
        self.observed = observation
        # set messages (won't change)
        n, q = self.outgoing.shape
        for i in range(n):
            self.outgoing[i] = np.zeros((1, self.dim))
            self.outgoing[i, self.observed] = 1.
        self.next_step()  # copy into oldoutgoing

    def prep_messages(self, p, adjacency):
        """
        Multiplies together incoming messages to make new outgoing.
        """

        # compute new messages if no observation has been made
        if self.enabled and self.observed < 0 and len(self.nbrs) > 1:
            # switch reference for old messages
            self.next_step()
            n, q = self.incoming.shape
            # create an auxiliary matrix for next step
            aux = self.incoming.dot(p)
            for i in range(n):
                # multiply together all excluding message at current index
                holder = aux[i, :]
                aux[i, :] = np.ones(q)
                arr = adjacency[i, :].reshape((-1, 1))
                filtering = np.hstack((arr, arr))
                self.outgoing[i, :] = np.prod(aux * filtering, axis=0)
                aux[i, :] = holder

            # normalize once finished with all messages
            self.normalize_messages()
