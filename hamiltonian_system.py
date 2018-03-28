import numpy as np

def state_energy_density(configuration, correlation_strength):
    n = configuration.shape
    state_energy = configuration.dot(correlation_strength).dot(configuration.T)
    return  - state_energy / n

def build_system_from_graph(adjacency, type):
    pass
