import brainpy.math as bm


def sigmoid(x):
    return 1 / (1 + bm.exp(-x))


def edge_states(num, sigma, geometry, edge_type='tanh'):
    """
    Return the canonical edge profile on the discrete neuron grid.
    """
    k0 = geometry.k0
    k1 = geometry.k1
    k2 = geometry.k2
    k = bm.arange(num)
    if edge_type == 'Laplace':
        return bm.exp(-bm.exp(sigma * (bm.pi / (k2 - k1)) * (k - k0)))
    elif edge_type == 'tanh':
        sigma_prime = 4 * sigma / bm.exp(1)
        return sigmoid(-sigma_prime * bm.pi / (k2 - k1) * (k - k0))
    else:
        raise ValueError('Edge type should be either Laplace or tanh.')


def bump_states(num, sigma, geometry):
    return bump_states_at_idx(num, sigma, geometry, geometry.k0)


def bump_states_at_idx(num, sigma, geometry, center_idx):
    k1 = geometry.k1
    k2 = geometry.k2
    k = bm.arange(num)
    return bm.exp(-(bm.pi / (bm.sqrt(2) * sigma * (k2 - k1))) ** 2 * (k - center_idx) ** 2)
