import brainpy as bp
import brainpy.math as bm


def sigmoid(x):
    return 1/(1 + bm.exp(-x))

def logit(x):
    return bm.log(x / (1 - x))


def _theta_grid(num):
    """theta(k) = -pi/2 + pi/num * k for k in [0, num-1]."""
    return -bm.pi / 2 + (bm.pi / num) * bm.arange(num)


def _validate_theta_params(theta_0, enc_range):
    if theta_0 is None:
        raise ValueError("theta_0 is required.")
    if enc_range is None or enc_range <= 0:
        raise ValueError("enc_range must be positive.")
    
def edge_states(num, sigma, theta_0, enc_range, edge_type='tanh'):
    """
    Calculate edge states on theta grid.
    theta(k) = -pi/2 + pi/num * k
    """
    _validate_theta_params(theta_0, enc_range)
    theta = _theta_grid(num)
    x = bm.pi / enc_range * (theta - theta_0)
    if edge_type == 'Laplace':
        return bm.exp(-bm.exp(sigma * x))
    elif edge_type == 'tanh':
        sigma_prime = 4 * sigma / bm.exp(1)
        return sigmoid(-sigma_prime * x)
    else:
        raise ValueError('Edge type should be either Laplace or tanh.')

def derivative_edge_states(num, sigma, theta_0, enc_range, edge_type='tanh'):
    """
    Calculate d/dtheta of the edge states profile.
    """
    _validate_theta_params(theta_0, enc_range)
    theta = _theta_grid(num)
    gain = bm.pi / enc_range
    x = gain * (theta - theta_0)

    if edge_type == 'Laplace':
        z = sigma * x
        return -sigma * gain * bm.exp(z) * bm.exp(-bm.exp(z))
    elif edge_type == 'tanh':
        sigma_prime = 4 * sigma / bm.exp(1)
        return -sigma_prime * gain * (1 - bm.square(bm.tanh(sigma_prime * x / 2))) / 4
    else:
        raise ValueError('Edge type should be either Laplace or tanh.')

def bump_states(num, sigma, theta_0, enc_range):
    _validate_theta_params(theta_0, enc_range)
    theta = _theta_grid(num)
    return bm.exp(-(bm.pi / (bm.sqrt(2) * sigma * enc_range)) ** 2 * (theta - theta_0) ** 2)

def DoG_kernel(num, sigma_E, s, offset=0):
    assert offset< bm.pi/2 and offset> -bm.pi/2, "Offset must be between -pi/2 and pi/2."
    theta = bm.linspace(-bm.pi/2, bm.pi/2, num)
    sigma_I = s * sigma_E
    G_E = 1/(bm.sqrt(2*bm.pi) * sigma_E) * bm.exp(-0.5 * ((theta - offset) / sigma_E) ** 2)
    G_I = 1/(bm.sqrt(2*bm.pi) * sigma_I) * bm.exp(-0.5 * ((theta - offset) / sigma_I) ** 2)
    return  (G_E - 1/s * G_I)

def Gaussian_kernel(num, sigma_E, offset=0):
    assert offset< bm.pi/2 and offset> -bm.pi/2, "Offset must be between -pi/2 and pi/2."
    theta = bm.linspace(-bm.pi/2, bm.pi/2, num)
    return 1/(bm.sqrt(2*bm.pi) * sigma_E) * bm.exp(-0.5 * ((theta - offset) / sigma_E) ** 2)

def make_conn_mat_from_kernel(num, kernel, normed=False):
    """
    Takes a 1D list specifying the shape of an interaction kernel and produces an
    interaction matrix for neurons lying along a 1D space.
    
    The kernel is assumed to be zero outside of the range [-N/2,N/2].

    
    num                  : the number of neurons 
    discreteKernel       : Should have an odd length <= N.  The kernel is centered on the
                            middle element (index (len(discreteKernel)-1)/2).
    N                    : The number of neurons (dimension of the output matrix is NxN)
    normed (True)        : If True, each row is normalized to sum to 1.
    """
    # Nk = len(Kernel)
    # assert(Nk%2==1) # discreteKernel should have odd length
    # assert(Nk <= self.num) # discreteKernel should have length less than or equal to N
    # mat = bm.zeros((self.num, self.num))
    
    # # copy in the appropriate part of the kernel for each row
    # for i in range(self.num):
    #     matIndexMin = max(0,i-(Nk-1)//2)
    #     matIndexMax = min(i+Nk-(Nk-1)//2, self.num)
    #     kIndexMin = 0 + max(0, matIndexMin - (i-(Nk-1)//2))
    #     kIndexMax = Nk - max(0,i+Nk-(Nk-1)//2 - self.num)
    #     mat[i,matIndexMin:matIndexMax] = Kernel[kIndexMin:kIndexMax]
    #     Nk = Kernel.size
    Nk = len(kernel)
    assert Nk % 2 == 1, "Kernel length must be odd."
    assert Nk <= num, "Kernel length must be less than or equal to N."
    
    m = (Nk - 1) // 2  # center index of the kernel
    # Create row and column indices for the N x N matrix.
    i = bm.arange(num).reshape(num, 1)
    j = bm.arange(num).reshape(1, num)
    
    # The relative offset from the kernel's center is (j-i)
    diff = j - i
    # The kernel index is diff shifted by m.
    indices = diff + m
    
    # Create a mask for valid indices (i.e., where the kernel is defined).
    valid = (indices >= 0) & (indices < Nk)
    
    # Build the connection matrix.
    conn = bm.zeros((num, num))
    conn[valid] = kernel[indices[valid]]
    if normed:
        # Normalize each row to sum to 1.
        row_sums = conn.sum(axis=1, keepdims=True)
        # Avoid division by zero.
        row_sums[row_sums == 0] = 1
        conn = conn / row_sums
    return conn

    
def make_bump_conn_mat(num, sigma, beta, clamp_frac=0.2, offset=0):
    """
    Build the bump-population connectivity matrix.
    """
    num_interior = int(num * (1 - clamp_frac))
    if num_interior % 2 == 0:
        num_interior = num_interior - 1
    kernel = Gaussian_kernel(num_interior, sigma, offset=offset)
    J_BB = make_conn_mat_from_kernel(num, kernel, normed=True)
    return J_BB
    

def make_edge_conn_mat(num, sigma, theta_0, enc_range, edge_type='tanh',
                       clamp_frac=0.2, offset=0, fit_half_width=None):
        """
        Build the edge-population connectivity matrix.
        """
        _validate_theta_params(theta_0, enc_range)
        # If optimize_offset=True, the passed-in offset is ignored and replaced by the searched value.
        if edge_type == 'tanh':
            sigma = 4 * sigma / bm.exp(1)
        def _base_kernel_matrix(clamp_frac, offset):
            """
            num_interior: the number of neurons in the interior of the edge population
            offset_value: the offset of the kernel
            """
            sigma_E = 1
            s = 1.2
            num_interior = int(num * (1 - clamp_frac))
            if num_interior % 2 == 0:
                num_interior = num_interior - 1
            kernel = DoG_kernel(num_interior, sigma_E, s, offset=offset)
            J_EE_local = make_conn_mat_from_kernel(num, kernel, normed=True)
            return J_EE_local

        def _match_linear_drive(J_EE_local):
            # Match linear drive in interior
            theta = _theta_grid(num)
            real_r = edge_states(
                num, sigma, edge_type=edge_type, theta_0=theta_0, enc_range=enc_range
            )
            U0 = J_EE_local @ real_r
            local_half_width = fit_half_width
            if local_half_width is None:
                local_half_width = 10 * bm.pi / num
            fit_mask = bm.abs(theta - theta_0) <= local_half_width
            th = theta[fit_mask]
            y = U0[fit_mask]
            A, B = bm.polyfit(th, y, deg=1)
            J0 = -sigma / A
            return J0
        J_EE = _base_kernel_matrix(clamp_frac, offset)
        #J_EE = J_EE - J_EE.mean(axis=1, keepdims=True)
        J0 = _match_linear_drive(J_EE)
        beta = - J0/2
        return J0 * J_EE, beta



# def find_optimal_edge_offset(num, sigma, theta_0, enc_range, edge_type='tanh',
#                              tol=1e-3, max_iter=500, progress=False):
#         """
#         Find the optimal edge offset for the edge population by searching over a range of offsets.
#         Uses a continuous ternary search (efficient for unimodal functions) to find the real-valued
#         offset that minimizes ||sigmoid(J_EE(offset) @ r_E0) - r_E0||.
#         If progress=True, show a tqdm progress bar for iterations.
#         """
#         # Searches offset only; uses make_edge_conn_mat with optimize_offset=False to avoid recursion.
#         # Define search range (fraction of total neurons). Tunable if needed.
#         search_frac = 0.5
#         left = -search_frac * (bm.pi / 2)
#         right = search_frac * (bm.pi / 2)

#         def _edge_error(offset):
#             J_EE_temp = make_edge_conn_mat(num, sigma, theta_0, enc_range,
#                                            edge_type=edge_type,
#                                            offset=offset,
#                                            optimize_offset=False)
            
#             r_0 = [edge_states(num, sigma, th, enc_range, edge_type=edge_type)
#                    for th in bm.linspace(theta_0 - enc_range / 2, theta_0 + enc_range / 2, 10)]
#             err = bm.max(bm.array([bm.sqrt(bm.sum(bm.square(sigmoid(J_EE_temp @ r) - r))) for r in r_0]))
#             try:
#                 return float(err)
#             except Exception:
#                 return float(bm.to_numpy(err))

#         def _ternary_search(left, right):
#             it = 0
#             pbar = None
#             if progress:
#                 pbar = tqdm(total=max_iter, desc='find_optimal_edge_offset')

#             while it < max_iter:
#                 m1 = left + (right - left) / 3.0
#                 m2 = right - (right - left) / 3.0
#                 err1 = _edge_error(m1)
#                 err2 = _edge_error(m2)
#                 if progress and pbar is not None:
#                     pbar.set_postfix({'left': left, 'right': right,
#                                       'err1': err1, 'err2': err2})
#                     pbar.update(1)
#                 if err1 > err2:
#                     left = m1
#                 else:
#                     right = m2
#                 if (err1+err2)/2 < tol:
#                     break
#                 it += 1

#             if pbar is not None:
#                 pbar.close()
#             return (left + right) / 2.0

#         best_offset = _ternary_search(left, right)
#         best_err = _edge_error(best_offset)
#         print(f"find_optimal_edge_offset: chosen offset {best_offset:.6f} with error {best_err:.6f}")
#         return best_offset