import numpy as np
import belief_propagation as bp
import regular_graph as reg
import propagation_graph as graph
import hamiltonian_system as hs
from extremal_optimization import EONode, EOGraph


def error_rate(result, ground_truth):
    ground_truth = ground_truth.ravel()
    n = ground_truth.shape
    err1 = np.linalg.norm(result - ground_truth, ord=0) / n
    err2 = np.linalg.norm(result + ground_truth, ord=0) / n
    return min(err1, err2)


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


# # for extremal optimization on SBM
# n = 500
# q = 2
# a = 10
# b = 2
# tau = 1.4
# A, ground_truth, p_in, p_out = bp.select_model("sparse-sbm", n, a, b)
# g = EOGraph(A, 2)
# g.construct_from_adj(hamiltonian=True)
# result = g.extremal_optimization(tau, hamiltonian=True)
# # print(result[: int(n / 2)])
# # print(result[int(n / 2) :])
# err_rate = error_rate(result, ground_truth)
# print("The error rate is: {}".format(err_rate))
# density = hs.state_energy_density(result, A)
# print("The state energy density is: {}".format(density))


# for extremal optimization on regular graph hamiltonian
n = 500
tau = 1.4
A = reg.regular_graph(n, 3)
# A becomes interaction_strength J
J = hs.build_system_from_graph(A, 'glasses')
g = EOGraph(J, 2)
g.construct_from_adj(hamiltonian=True)
result = g.extremal_optimization(tau, hamiltonian=True)
density = hs.state_energy_density(result, J)
print("The state energy density is: {}".format(density))

g_old = EOGraph(J, 2)
g_old.construct_from_adj(hamiltonian=True)
result = g_old.extremal_optimization(tau, hamiltonian=True, version='old')
density = hs.state_energy_density(result, J)
print("The state energy density is: {}".format(density))
