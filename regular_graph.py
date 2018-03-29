import numpy as np
import networkx as nx


def regular_graph(n, degree):
    g = nx.random_regular_graph(degree, n)
    A = nx.to_numpy_matrix(g)
    return np.array(A)
