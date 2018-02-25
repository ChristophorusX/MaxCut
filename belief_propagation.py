import numpy as np
import sbm_generator as sbm


def get_init_phi(n, q):
    """
    Returns a random initialization of all the messages of each pair of
    vertices given they are in the same group or not.

    Parameters
    ----------
    n : int
        Number of vertices.
    q : int
        Number of clusters.

    Returns
    -------
    phi
        Random initialization of message tensor phi.

    """

    phi = np.random.uniform(0, 1, (n, n, q))
    phi_empty = np.zeros((n, n, q))

    phi = normalize(phi)

    return phi, phi_empty


def normalize(phi):
    """
    Returns a new tensor phi with probablities been normalized.
    """

    n, _, q = phi.shape
    phi_norm = np.sum(phi, axis=2)
    for r in range(q):
        phi[:, :, r] = divide_safezero(phi[:, :, r], phi_norm)
    return phi


def edge_prob_mat(p_in, p_out):
    """
    Returns the matrix containing edge probablities for vector product.
    """

    return np.array([[p_in, p_out], [p_out, p_in]])


def select_model(model, n, a, b):
    """
    Generates SBM observations from selected regime.
    """

    if model == "sparse-sbm":
        A, ground_truth = sbm.sbm_linear(n, a, b)
        p_in = a / n
        p_out = b / n
        return A, ground_truth, p_in, p_out
    if model == "log-sbm":
        A, ground_truth = sbm.sbm_logarithm(n, a, b)
        p_in = a * np.log(n) / n
        p_out = b * np.log(n) / n
        return A, ground_truth, p_in, p_out
    if model == "const-sbm":
        A, ground_truth = sbm._stochastic_block_model(n, a, b)
        p_in = a
        p_out = b
        return A, ground_truth, p_in, p_out
    else:
        return None, None, None, None


def belief_propagation(A, phi, phi_empty, p, n_repeat):
    """
    Belief propagation algorithm on observation A that returns a propogated
    tensor phi.
    """

    n, _, q = phi.shape
    for iteration in range(n_repeat):
        for i in range(n):
            for j in range(n):
                for r in range(q):
                    prod = 1
                    for k in range(n):
                        if A[i, k] == 1 and k != j:
                            summation = phi[k, i, :].dot(p[r, :])
                            prod *= summation
                    phi_empty[i, j, r] = prod
        phi = phi_empty[:, :, :]

        phi = normalize(phi)

    return phi


def marginal_prob(A, phi, p):
    """
    Returns a matrix of marginal probability of being in some cluster
    for each vertex.
    """

    n, _, q = phi.shape
    marginal = np.empty((n, q))

    for i in range(n):
        for r in range(q):
            prod = 1
            for j in range(n):
                if A[i, j] == 1:
                    summation = phi[j, i, :].dot(p[r, :])
                    prod *= summation
            marginal[i, r] = prod

    marginal_norm = np.sum(marginal, axis=1)
    for r in range(q):
        # marginal[:, r] = marginal[:, r] / marginal_norm
        marginal[:, r] = divide_safezero(marginal[:, r], marginal_norm)

    return marginal


def divide_safezero(a, b):
    '''
    Divies a by b, then turns nans and infs into 0, so all division by 0
    becomes 0.
    Args:
        a (np.ndarray)
        b (np.ndarray|int|float)
    Returns:
        np.ndarray
    '''
    # deal with divide-by-zero: turn x/0 (inf) into 0, and turn 0/0 (nan) into
    # 0.
    c = a / b
    c[c == np.inf] = 0.0
    c = np.nan_to_num(c)
    return c


if __name__ == "__main__":
    n = 10
    q = 2
    a = .9
    b = .1
    A, ground_truth, p_in, p_out = select_model("const-sbm", n, a, b)
    # print("The observation is: \n{}".format(A))
    p = edge_prob_mat(p_in, p_out)

    phi_init, phi_empty = get_init_phi(n, q)
    # print("Init Phi is: \n{}".format(phi_init[:, :, 0]))
    phi = belief_propagation(A, phi_init, phi_empty, p, 100)
    marginal = marginal_prob(A, phi, p)

    print("The propagation result is: \n{}".format(phi[:, :, 0]))
    print("The marginal from BP is: \n{}".format(marginal))
    print("The ground truth is: \n{}".format(ground_truth))
