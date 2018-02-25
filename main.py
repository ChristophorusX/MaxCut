import numpy as np
import belief_propagation as bp
import propagation_graph as graph


n = 10
q = 2
a = .9
b = .1
A, ground_truth, p_in, p_out = bp.select_model("const-sbm", n, a, b)
p = bp.edge_prob_mat(p_in, p_out)
