import numpy as np
import belief_propagation as bp
import propagation_graph as graph
from extremal_optimization import EONode, EOGraph

# # for belief propagation
# n = 50
# q = 2
# a = .2
# b = .2
# A, ground_truth, p_in, p_out = bp.select_model("const-sbm", n, a, b)
# p = bp.edge_prob_mat(1, 0)
# g = graph.Graph(A, 2, p)
# g.construct_from_adj()
# marginals = g.marginals(maxsteps=100)
# print("MARGINALS:\n{}".format(marginals))

# for extremal optimization
n = 500
q = 2
a = 3
b = 10
tau = 1.4
A, ground_truth, p_in, p_out = bp.select_model("sparse-sbm", n, a, b)
p = bp.edge_prob_mat(p_in, p_out)
g = EOGraph(A, 2, p)
g.construct_from_adj()
result = g.extremal_optimization(tau)
print(result)
