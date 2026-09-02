import brainpy.math as bm


def sigmoid(x):
    return 1 / (1 + bm.exp(-x))


def theta_grid(num, geometry):
    if num <= 1:
        return bm.asarray([geometry.theta_min])
    return bm.linspace(geometry.theta_min, geometry.theta_max, num)

def edge_states(num, gamma, geometry, edge_type='tanh', center_pos=0.0):
    """
    Return the canonical edge profile on the discrete neuron grid.
    """
    theta = theta_grid(num, geometry)
    theta_rel = theta - center_pos
    if edge_type == 'Laplace':
        return bm.exp(-bm.exp(gamma * theta_rel))
    elif edge_type == 'tanh':
        gamma_sigma = 4 * gamma / bm.exp(1)
        return sigmoid(-gamma_sigma * theta_rel)
    else:
        raise ValueError('Edge type should be either Laplace or tanh.')


def bump_states(num, sigma, geometry, center_pos=0.0):
    theta = theta_grid(num, geometry)
    theta_rel = theta - center_pos
    return bm.exp(-0.5 * (theta_rel / sigma) ** 2)
