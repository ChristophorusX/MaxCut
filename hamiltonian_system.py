import numpy as np


def state_energy_density(configuration, interaction_strength):
    n = configuration.shape
    state_energy = configuration.dot(
        interaction_strength).dot(configuration.T) / 2
    return - state_energy / n


def build_system_from_graph(adjacency, type):
    n, _ = adjacency.shape
    if type == 'gaussian':
        random = np.random.normal(size=(n, n))
        interaction_strength = adjacency * random
        return interaction_strength
    if type == 'glasses':
        random = np.random.randint(2, size=(n, n)) * 2 - 1
        interaction_strength = adjacency * random
        return interaction_strength
