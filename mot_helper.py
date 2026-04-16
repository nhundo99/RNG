import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

def evaluate_dual(x, y, A, B, u, C, V, r, c, W, eta, eps):
    """
    Dual objective f(x,y,A,B,u).
    Returns -inf if the transport exponential would overflow.
    """
    Z = eta * (-C + A @ V.T + B @ V.T + x[:, None] + y[None, :]) - 1.0

    if not np.all(np.isfinite(Z)):
        return -np.inf
    if np.max(Z) > 700:
        return -np.inf

    P = np.exp(Z)

    val = np.sum(x * r) + np.sum(y * c) - np.sum(P) / eta
    val += np.sum(A * W) + np.sum(B * W)
    val -= np.sum(np.exp(eta * A - 1.0)) / eta
    val -= np.sum(np.exp(-eta * B - 1.0)) / eta
    val -= np.sum(np.exp(eta * (u - A + B) - 1.0)) / eta
    val += eps * u - np.exp(eta * u - 1.0) / eta

    return val

def compute_P(x, y, A, B, C, V, eta=1.0):
    """
    Compute transport plan P from dual variables.
    x, y: (n,)
    A, B: (n, d)
    C   : (n, n)
    V   : (n, d) rows v_j
    """
    AV = A @ V.T
    BV = B @ V.T
    X = x[:, None]
    Y = y[None, :]
    Z = eta * (-C + AV + BV + X + Y) - 1.0
    
    P = np.exp(Z)
    return P

def update_y(P, c, y, eta=1.0):
    """
    Column scaling: enforce column sums ~ c.
    """
    col_sums = P.sum(axis=0)
    y_new = y + (np.log(c) - np.log(col_sums))/eta
    return y_new

def grad_g(x, A, B, u, P, V, r, W, eta=1.0, eps=1.0):
    """
    Exact gradient of full dual wrt g = (x,A,B,u).
    """

    n, d = A.shape

    # Transport statistics
    row_sums = P.sum(axis=1)      # (n,)
    Pv = P @ V                   # (n,d)

    # x-gradient
    grad_x = r - row_sums

    # Exponentials for entropy terms
    exp_A = np.exp(eta * A - 1.0)
    exp_B = np.exp(-eta * B - 1.0)
    exp_cpl = np.exp(eta * (u - A + B) - 1.0)

    # A-gradient
    grad_A = -Pv + W - exp_A + exp_cpl

    # B-gradient
    grad_B = -Pv + W + exp_B - exp_cpl

    # u-gradient
    grad_u = (
        eps
        - np.exp(eta * u - 1.0)
        - exp_cpl.sum()
    )

    return grad_x, grad_A, grad_B, grad_u

def hessian_g(x, A, B, u, P, V, eta=1.0):
    n, d = A.shape

    row_sums = P.sum(axis=1)
    Pv_sq = P @ (V**2)

    exp_A = np.exp(eta * A - 1.0)
    exp_B = np.exp(-eta * B - 1.0)
    exp_cpl = np.exp(eta * (u - A + B) - 1.0)

    # 1. Main Diagonal Blocks
    H_xx = sparse.diags(-eta * row_sums)
    H_AA = sparse.diags((-eta * Pv_sq - eta * exp_A - eta * exp_cpl).ravel())
    H_BB = sparse.diags((-eta * Pv_sq - eta * exp_B - eta * exp_cpl).ravel())

    # 2. Cross terms between A and B
    diag_AB = (-eta * Pv_sq + eta * exp_cpl).ravel()
    H_AB = sparse.diags(diag_AB)
    H_BA = H_AB  # Symmetric

    # 3. Cross terms between x and A/B
    row_idx = np.repeat(np.arange(n), d)
    col_idx = np.arange(n * d)
    H_xA_data = -eta * (P @ V).ravel()
    
    H_xA = sparse.coo_matrix((H_xA_data, (row_idx, col_idx)), shape=(n, n*d))
    H_Ax = H_xA.T
    
    H_xB = sparse.coo_matrix((H_xA_data, (row_idx, col_idx)), shape=(n, n*d))
    H_Bx = H_xB.T

    # 4. u terms (Dense borders appended as sparse vectors)
    H_uu = sparse.csr_matrix([[-eta * np.exp(eta * u - 1.0) - eta * exp_cpl.sum()]])
    
    H_uA = sparse.csr_matrix((eta * exp_cpl).ravel())
    H_Au = H_uA.T
    
    H_uB = sparse.csr_matrix((-eta * exp_cpl).ravel())
    H_Bu = H_uB.T
    
    H_ux = sparse.csr_matrix((1, n))
    H_xu = H_ux.T

    # Assemble the full block arrow-head matrix
    H = sparse.bmat([
        [H_xx, H_xA, H_xB, H_xu],
        [H_Ax, H_AA, H_AB, H_Au],
        [H_Bx, H_BA, H_BB, H_Bu],
        [H_ux, H_uA, H_uB, H_uu]
    ], format='csc')

    return H

def newton_step_g(x, A, B, u, P, V, r, W, eta=1.0, eps=1.0):
    """
    Newton step for g = (x, A, B, u) in the simplified dual.
    """
    n, d = A.shape

    # Get Gradients
    grad_x, grad_A, grad_B, grad_u = grad_g(x, A, B, u, P, V, r, W, eta=eta, eps=eps)

    # Build the full approximate sparse Hessian
    H = hessian_g(x, A, B, u, P, V, eta=eta)

    # Stack gradients: [x; vec(A); vec(B); u]
    grad_vec_g = np.concatenate([
        grad_x,
        grad_A.ravel(),
        grad_B.ravel(),
        [grad_u]
    ])

    # Solve H * delta = -grad (Newton step for maximization of concave f)
    # Use negative damping because H is safely negative definite
    damping = 1e-8
    H_damped = H - damping * sparse.eye(H.shape[0])

    delta = spsolve(H_damped, -grad_vec_g)

    # Unstack the updated delta
    dx = delta[:n]
    dA = delta[n : n + n*d].reshape(n, d)
    dB = delta[n + n*d : n + 2*n*d].reshape(n, d)
    du = delta[-1]

    return dx, dA, dB, du, grad_x, grad_A, grad_B, grad_u

def line_search_armijo(f_eval, x, A, B, u, dx, dA, dB, du, gx, gA, gB, gu,
                       y, C, V, r, c, W, eta, eps,
                       alpha_init=1.0, beta=0.5, gamma=1e-4, alpha_min=1e-8):
    """
    Backtracking line search using the Armijo condition for maximization.
    gx, gA, gB, gu are the gradients of the objective function at the current step.
    """
    alpha = alpha_init
    f0 = f_eval(x, y, A, B, u, C, V, r, c, W, eta, eps)

    # Compute the directional derivative: <grad f, delta g>
    # np.sum() across element-wise multiplication acts as the correct inner product for matrices.
    dir_deriv = (np.sum(gx * dx) + 
                 np.sum(gA * dA) + 
                 np.sum(gB * dB) + 
                 np.sum(gu * du))

    # Safety check: if dir_deriv <= 0, the Newton step is not an ascent direction.
    if dir_deriv <= 0:
        return 0.0

    while alpha > alpha_min:
        x_new = x + alpha * dx
        A_new = A + alpha * dA
        B_new = B + alpha * dB
        u_new = u + alpha * du

        # Guard against NaNs or Infs during extreme steps
        if not (np.all(np.isfinite(x_new)) and
                np.all(np.isfinite(A_new)) and
                np.all(np.isfinite(B_new)) and
                np.all(np.isfinite(u_new))):
            alpha *= beta
            continue

        f_new = f_eval(x_new, y, A_new, B_new, u_new, C, V, r, c, W, eta, eps)

        # Armijo condition for MAXIMIZATION
        # We want sufficient increase proportional to the directional derivative.
        if np.isfinite(f_new) and f_new >= f0 + gamma * alpha * dir_deriv:
            return alpha

        alpha *= beta

    return 0.0

def line_search_armijo_full(f_eval, x, y, A, B, u, 
                            dx, dy, dA, dB, du, 
                            gx, gy, gA, gB, gu, 
                            C, V, r, c, W, eta, eps,
                            alpha_init=1.0, beta=0.5, gamma=1e-4, alpha_min=1e-6):
    """
    Backtracking line search using the Armijo condition for the full Sinkhorn-Newton-Sparse step.
    Includes the y variable.
    """
    alpha = alpha_init
    f_curr = f_eval(x, y, A, B, u, C, V, r, c, W, eta, eps)

    # Compute the full directional derivative including y
    dir_deriv = (np.sum(gx * dx) + 
                 np.sum(gy * dy) + 
                 np.sum(gA * dA) + 
                 np.sum(gB * dB) + 
                 np.sum(gu * du))

    if dir_deriv <= 0:
        return 0.0  # Not an ascent direction

    while alpha > alpha_min:
        x_new = x + alpha * dx
        y_new = y + alpha * dy
        A_new = A + alpha * dA
        B_new = B + alpha * dB
        u_new = u + alpha * du

        # Finiteness check
        if not (np.all(np.isfinite(x_new)) and np.all(np.isfinite(y_new)) and
                np.all(np.isfinite(A_new)) and np.all(np.isfinite(B_new)) and
                np.isfinite(u_new)):
            alpha *= beta
            continue

        f_new = f_eval(x_new, y_new, A_new, B_new, u_new, C, V, r, c, W, eta, eps)

        # Armijo condition for MAXIMIZATION
        if np.isfinite(f_new) and f_new >= f_curr + gamma * alpha * dir_deriv:
            return alpha
            
        alpha *= beta
        
    return 0.0


def sinkhorn_type_MOT(C, r, c, V, W,
                      N_outer=50, Ng=1, eta=1.0, eps=1.0,
                      f_eval=evaluate_dual, track_history=True):
    """
    Algorithm 1: Sinkhorn-type entropic MOT.
    C: (n,n) cost matrix
    r,c: (n,) marginals
    V: (n,d), W: (n,d) encode martingale constraint PV ≈ W
    """
    n = C.shape[0]
    d = V.shape[1]

    x = np.zeros(n)
    y = np.zeros(n)
    A = np.zeros((n, d))
    B = np.zeros((n, d))
    u = 0.0

    history = {
        'iteration': [], 'time': [], 'martingale_violation': [],
        'row_violation': [], 'col_violation': [], 'transport_cost': [], 'p_diff': [], 'dual': []
    }

    start_time = time.time()
    P_prev = None
    

    for it in range(N_outer):
        # Step 1: form P
        P = compute_P(x, y, A, B, C, V, eta=eta)

        # Step 2: column scaling (update y)
        y = update_y(P, c, y, eta=eta)

        # Step 3: Newton steps on g = (x,A,B,u)
        for ig in range(Ng):
            dx, dA, dB, du, gx, gA, gB, gu = newton_step_g(
                x, A, B, u, P, V, r, W, eta=eta, eps=eps
            )

            alpha = line_search_armijo(
                f_eval, x, A, B, u,
                dx, dA, dB, du,
                gx, gA, gB, gu, 
                y, C, V, r, c, W, eta, eps
            )

            if alpha == 0.0:
                print(f"Line search failed at outer {it}, inner {ig}; stopping.")
                break

            x = x + alpha * dx
            A = A + alpha * dA
            B = B + alpha * dB
            u = u + alpha * du

        if track_history:
            P_current = compute_P(x, y, A, B, C, V, eta=eta)

            if not np.all(np.isfinite(P_current)):
                print("Non-finite P in history block; stopping.")
                break

            current_time = time.time() - start_time
            row_viol = np.max(np.abs(np.sum(P_current, axis=1) - r.flatten()))
            col_viol = np.max(np.abs(np.sum(P_current, axis=0) - c.flatten()))
            martingale_viol = np.max(np.abs(P_current @ V - W))
            cost = np.sum(P_current * C)
            p_diff = np.sum(np.abs(P_current - P_prev)) if P_prev is not None else np.nan
            P_prev = P_current.copy()
            f = evaluate_dual(x, y, A, B, u, C, V, r, c, W, eta, eps)

            history['iteration'].append(it)
            history['time'].append(current_time)
            history['row_violation'].append(row_viol)
            history['col_violation'].append(col_viol)
            history['martingale_violation'].append(martingale_viol)
            history['transport_cost'].append(cost)
            history['p_diff'].append(p_diff)
            history['dual'].append(f)

    P = compute_P(x, y, A, B, C, V, eta=eta)
    return P, (x, y, A, B, u), history

# --- Sparse Newton phase (Algorithm 2) ---

def sparsify_P(P, keep_frac=0.1):
    """
    Keep largest entries of P in absolute value; zero out rest.
    """
    n = P.shape[0]
    k = int(keep_frac * n * n)
    flat = P.ravel()
    if k <= 0:
        return np.zeros_like(P)
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[idx] = True
    P_sparse = np.zeros_like(flat)
    P_sparse[mask] = flat[mask]
    return P_sparse.reshape(P.shape)

def full_sparse_hessian(x, y, A, B, u, P_dense, P_sp, V, eta=1.0):
    """
    Constructs the full joint Hessian for z = (y, x, A, B, u).
    Uses P_dense for exact diagonal/block-diagonal curvature, 
    and P_sp for off-diagonal cross-derivatives to maintain O(n^2) complexity.
    """
    n, d = A.shape
    
    # --- EXACT COMPONENTS (Using P_dense) ---
    row_sums = np.array(P_dense.sum(axis=1)).flatten()
    col_sums = np.array(P_dense.sum(axis=0)).flatten()
    
    Pv = P_dense @ V
    Pv_sq = P_dense @ (V**2)
    
    exp_A = np.exp(eta * A - 1.0)
    exp_B = np.exp(-eta * B - 1.0)
    exp_cpl = np.exp(eta * (u - A + B) - 1.0)
    
    # ---------------------------------------------------------
    # 1. The y-blocks 
    # ---------------------------------------------------------
    # EXACT diagonal for y
    H_yy = sparse.diags(-eta * col_sums)
    
    # SPARSE cross-terms using P_sp
    H_yx = -eta * P_sp.T
    H_xy = H_yx.T
    
    # Construct y-A cross derivatives explicitly from the sparse coordinates
    P_coo = P_sp.tocoo() if sparse.issparse(P_sp) else sparse.coo_matrix(P_sp)
    r_P, c_P, d_P = P_coo.row, P_coo.col, P_coo.data
    
    rows_y = np.tile(c_P, d)
    cols_A = np.concatenate([r_P * d + k for k in range(d)])
    data_yA = np.concatenate([-eta * d_P * V[c_P, k] for k in range(d)])
    
    H_yA = sparse.coo_matrix((data_yA, (rows_y, cols_A)), shape=(n, n*d))
    H_Ay = H_yA.T
    
    H_yB = H_yA.copy() 
    H_By = H_yB.T
    
    H_yu = sparse.csr_matrix((n, 1))
    H_uy = H_yu.T
    
    # ---------------------------------------------------------
    # 2. The g-blocks (EXACT curvature using P_dense)
    # ---------------------------------------------------------
    H_xx = sparse.diags(-eta * row_sums)
    H_AA = sparse.diags((-eta * Pv_sq - eta * exp_A - eta * exp_cpl).ravel())
    H_BB = sparse.diags((-eta * Pv_sq - eta * exp_B - eta * exp_cpl).ravel())
    H_AB = sparse.diags((-eta * Pv_sq + eta * exp_cpl).ravel())
    H_BA = H_AB
    
    row_idx = np.repeat(np.arange(n), d)
    col_idx = np.arange(n * d)
    H_xA_data = -eta * Pv.ravel()
    
    H_xA = sparse.coo_matrix((H_xA_data, (row_idx, col_idx)), shape=(n, n*d))
    H_Ax = H_xA.T
    H_xB = sparse.coo_matrix((H_xA_data, (row_idx, col_idx)), shape=(n, n*d))
    H_Bx = H_xB.T
    
    H_uu = sparse.csr_matrix([[-eta * np.exp(eta * u - 1.0) - eta * exp_cpl.sum()]])
    H_uA = sparse.csr_matrix((eta * exp_cpl).ravel())
    H_Au = H_uA.T
    H_uB = sparse.csr_matrix((-eta * exp_cpl).ravel())
    H_Bu = H_uB.T
    H_ux = sparse.csr_matrix((1, n))
    H_xu = H_ux.T
    
    # Assemble the massive joint block matrix
    H = sparse.bmat([
        [H_yy, H_yx, H_yA, H_yB, H_yu],
        [H_xy, H_xx, H_xA, H_xB, H_xu],
        [H_Ay, H_Ax, H_AA, H_AB, H_Au],
        [H_By, H_Bx, H_BA, H_BB, H_Bu],
        [H_uy, H_ux, H_uA, H_uB, H_uu]
    ], format='csc')
    
    return H

def sparse_newton_MOT(C, r, c, V, W, N1=50, N2=10, keep_frac=0.1, eps=1.0, eta=1.0):
    """
    Algorithm 2: Sinkhorn-Newton-Sparse (Joint Implementation with Stabilized Line Search).
    """
    # 1. Warm start using Sinkhorn-type MOT (Algorithm 1)
    P_current, (x, y, A, B, u), history = sinkhorn_type_MOT(
        C, r, c, V, W, N_outer=N1, eta=eta, eps=eps, track_history=True
    )
    
    n, d = A.shape
    P_prev = P_current.copy()
    
    t0 = time.time()
    last_time = history['time'][-1] if history['time'] else 0.0
    
    # 2. Sparse Newton Phase
    for it in range(N1, N1 + N2):
        P_dense = compute_P(x, y, A, B, C, V, eta=eta)
        P_sp = sparsify_P(P_dense, keep_frac=keep_frac)
        
        # Exact gradients
        grad_y = c.flatten() - P_dense.sum(axis=0)
        grad_x, grad_A, grad_B, grad_u = grad_g(x, A, B, u, P_dense, V, r, W, eta=eta, eps=eps)
        grad_vec = np.concatenate([grad_y, grad_x, grad_A.ravel(), grad_B.ravel(), [grad_u]])
        
        # Approximate Sparse Hessian 
        H_sp = full_sparse_hessian(x, y, A, B, u, P_dense, P_sp, V, eta=eta)
        
        # Damping
        diag_max = np.max(np.abs(H_sp.diagonal()))
        damping = 1e-5 * diag_max + 1e-8
        H_damped = H_sp - damping * sparse.eye(H_sp.shape[0])

        delta = spsolve(H_damped, -grad_vec)
        
        # If solver fails entirely, break gracefully
        if np.any(np.isnan(delta)):
            print(f"Solver returned NaN at iter {it}. Stopping Newton phase.")
            break
        
        dy = delta[:n]
        dx = delta[n : 2*n]
        dA = delta[2*n : 2*n + n*d].reshape(n, d)
        dB = delta[2*n + n*d : 2*n + 2*n*d].reshape(n, d)
        du = delta[-1]
        
        # --- FIX 2: Line Search on Dual Objective ---
        alpha = line_search_armijo_full(
            evaluate_dual, x, y, A, B, u,
            dx, dy, dA, dB, du,
            grad_x, grad_y, grad_A, grad_B, grad_u,
            C, V, r, c, W, eta, eps
        )

        if alpha > 0.0:
            x += alpha * dx
            y += alpha * dy
            A += alpha * dA
            B += alpha * dB
            u += alpha * du
        else:
            print(f"Line search reached limit at iter {it}. Optimal point reached or stuck.")
            break
            
        # Recompute exact P after variables update
        P_current = compute_P(x, y, A, B, C, V, eta=eta)
        
        # --- History Tracking ---
        row_viol = np.max(np.abs(np.sum(P_current, axis=1) - r.flatten()))
        col_viol = np.max(np.abs(np.sum(P_current, axis=0) - c.flatten()))
        martingale_viol = np.max(np.abs(P_current @ V - W))
        cost = np.sum(P_current * C)
        p_diff = np.sum(np.abs(P_current - P_prev))
        P_prev = P_current.copy()
        f = evaluate_dual(x, y, A, B, u, C, V, r, c, W, eta, eps)
        
        history['iteration'].append(it)
        history['time'].append(last_time + (time.time() - t0))
        history['row_violation'].append(row_viol)
        history['col_violation'].append(col_viol)
        history['martingale_violation'].append(martingale_viol)
        history['transport_cost'].append(cost)
        history['p_diff'].append(p_diff)
        history['dual'].append(f)

    return P_current, (x, y, A, B, u), history

def plot_mot_coupling(P, K1, K2, r, c, T1, T2):
    """
    Plots the MOT transport plan (coupling matrix P) along with 
    the source (T1) and target (T2) marginal densities on shared, squared axes.
    Pads the densities AND the coupling matrix to the global boundaries.
    """
    # 1. Sync the Asset Price Domains (X and Y axes of the heatmap)
    price_min = min(np.min(K1), np.min(K2))
    price_max = max(np.max(K1), np.max(K2))
    
    # 2. Pad grids and densities with 0.0 at the global boundaries
    K1_pad = np.concatenate(([price_min], K1, [price_max]))
    r_pad = np.concatenate(([0.0], r, [0.0]))
    
    K2_pad = np.concatenate(([price_min], K2, [price_max]))
    c_pad = np.concatenate(([0.0], c, [0.0]))
    
    # 3. Pad the Coupling Matrix P with zeros to match the new grid dimensions
    # pad_width=((top, bottom), (left, right))
    P_pad = np.pad(P, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0.0)
    
    # 4. Sync the Density Amplitudes (for the marginal plots)
    max_density = max(np.max(r), np.max(c))
    density_limit = max_density * 1.05  # Add 5% padding
    
    # Create a 2x2 grid for the subplots
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 4, 0.2], height_ratios=[4, 1, 0.2])
    
    # --- Main Heatmap (Coupling Matrix P) ---
    ax_joint = plt.subplot(gs[0, 1])
    
    # Use the PADDED grids and PADDED matrix here
    mesh = ax_joint.pcolormesh(K2_pad, K1_pad, P_pad, cmap='viridis', shading='auto', vmax=np.percentile(P, 99.5))
    ax_joint.set_title("Optimal Transport Plan (Coupling P)")
    ax_joint.tick_params(labelbottom=False, labelleft=False) 
    
    # FORCE the heatmap to use the globally shared price limits
    ax_joint.set_xlim(price_min, price_max)
    ax_joint.set_ylim(price_min, price_max)
    
    # Add a diagonal line to show where S_T1 = S_T2
    ax_joint.plot([price_min, price_max], [price_min, price_max], color='white', linestyle='--', alpha=0.6, label='S_T1 = S_T2')
    ax_joint.legend(loc='upper left')

    # --- Bottom Marginal (Target Distribution T2, c) ---
    ax_marg_x = plt.subplot(gs[1, 1], sharex=ax_joint)
    ax_marg_x.plot(K2_pad, c_pad, color='darkorange', linewidth=2)
    ax_marg_x.fill_between(K2_pad, c_pad, alpha=0.3, color='darkorange')
    ax_marg_x.set_xlabel(f"Asset Price at T2 = {T2:.2f}")
    ax_marg_x.set_ylabel("Density")
    ax_marg_x.set_ylim(density_limit, 0) 
    
    # --- Left Marginal (Source Distribution T1, r) ---
    ax_marg_y = plt.subplot(gs[0, 0], sharey=ax_joint)
    ax_marg_y.plot(r_pad, K1_pad, color='teal', linewidth=2)
    ax_marg_y.fill_betweenx(K1_pad, r_pad, alpha=0.3, color='teal')
    ax_marg_y.set_ylabel(f"Asset Price at T1 = {T1:.2f}")
    ax_marg_y.set_xlabel("Density")
    ax_marg_y.set_xlim(density_limit, 0) 

    # --- Colorbar ---
    ax_cbar = plt.subplot(gs[0, 2])
    plt.colorbar(mesh, cax=ax_cbar, label='Joint Probability Mass')
    
    plt.tight_layout()
    plt.show()