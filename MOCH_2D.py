import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from dataclasses import dataclass

# =============================================================================
# 2D Population balance model solved by
# 1) the Method of Characteristics (MOCH)
# 2) a finite-volume (FV) discretization on an arbitrary grid
# =============================================================================

@dataclass(frozen=True)
class ModelParameters:
    """
    Container for all model and simulation parameters in 2D.
    """
    growth_constant: float = 0.02
    gamma: float = 0.005
    p: float = 1.0
    
    mean_L: float = 90.0
    std_L: float = 8.0
    dL: float = 1.5
    
    mean_W: float = 150.0
    std_W: float = 8.0
    dW: float = 1.5
    
    total_number: float = 5000.0
    final_time_min: float = 100.0
    n_time_points: int = 10

def compute_1d_effective_widths(X: np.ndarray) -> np.ndarray:
    """
    Compute 1D effective cell widths for an arbitrary characteristic grid vector.
    Valid for both equidistant and non-equidistant grids.
    """
    widths = np.zeros_like(X)
    if len(X) > 1:
        widths[1:-1] = 0.5 * (X[2:] - X[:-2])
        widths[0] = X[1] - X[0]
        widths[-1] = X[-1] - X[-2]
    return widths

def create_initial_grid(params: ModelParameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the initial 2D size grid spanning 4 standard deviations.
    """
    span_L = 4.0 * params.std_L
    start_L = params.mean_L - span_L
    stop_L = params.mean_L + span_L
    L_vec = np.arange(start_L, stop_L + 0.5 * params.dL, params.dL)

    span_W = 4.0 * params.std_W
    start_W = params.mean_W - span_W
    stop_W = params.mean_W + span_W
    W_vec = np.arange(start_W, stop_W + 0.5 * params.dW, params.dW)

    LL, WW = np.meshgrid(L_vec, W_vec, indexing='ij')
    return LL, WW

def create_extended_fv_grid(params: ModelParameters, max_L: float = 600.0, max_W: float = 600.0):
    """
    Construct an extended stationary 2D grid for the Eulerian finite-volume method.
    Calculates localized widths dynamically to inherently support non-equidistant grids.
    """
    start_L = params.mean_L - 4.0 * params.std_L
    start_W = params.mean_W - 4.0 * params.std_W
    
    L_vec = np.arange(start_L, max_L + 0.5 * params.dL, params.dL)
    W_vec = np.arange(start_W, max_W + 0.5 * params.dW, params.dW)
    
    delta_L = compute_1d_effective_widths(L_vec)
    delta_W = compute_1d_effective_widths(W_vec)
    
    LL_grid, WW_grid = np.meshgrid(L_vec, W_vec, indexing='ij')
    f_initial_fv = create_initial_number_density(LL_grid, WW_grid, params, delta_L, delta_W)
    
    return LL_grid, WW_grid, delta_L, delta_W, f_initial_fv

def create_initial_number_density(LL: np.ndarray, WW: np.ndarray, params: ModelParameters, 
                                  delta_L: np.ndarray = None, delta_W: np.ndarray = None) -> np.ndarray:
    """
    Construct the initial 2D Gaussian number density and normalize it.
    Uses precise area integration allowing for non-equidistant cells.
    """
    raw_density = norm.pdf(LL, loc=params.mean_L, scale=params.std_L) * \
                  norm.pdf(WW, loc=params.mean_W, scale=params.std_W)
    
    if delta_L is None or delta_W is None:
        area_matrix = params.dL * params.dW
    else:
        area_matrix = delta_L[:, None] * delta_W[None, :]
        
    integral_approx = np.sum(raw_density * area_matrix)
    f0 = raw_density / integral_approx * params.total_number
    return f0

def build_initial_state(LL: np.ndarray, WW: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """
    Assemble the ODE initial state vector for 2D.
    State ordering is defined as: y = [f, L, W]
    """
    return np.concatenate([f0.flatten(), LL.flatten(), WW.flatten()])

def growth_model_rhs(
    t: float,
    y: np.ndarray,
    growth_constant: float,
    gamma: float,
    p: float,
    n_total: int
) -> np.ndarray:
    """
    Right-hand side of the ODE system for the 2D method of characteristics.
    """
    f = y[:n_total]
    L = y[n_total:2*n_total]
    W = y[2*n_total:]

    dL_dt = growth_constant * (1.0 + gamma * L) ** p
    dW_dt = growth_constant * (1.0 + gamma * W) ** p

    div_v = growth_constant * gamma * p * ((1.0 + gamma * L) ** (p - 1.0) + (1.0 + gamma * W) ** (p - 1.0))
    df_dt = -f * div_v

    return np.concatenate([df_dt, dL_dt, dW_dt])

def fv_model_rhs(
    t: float,
    N_flat: np.ndarray,
    growth_constant: float,
    gamma: float,
    p: float,
    L_vec: np.ndarray,
    W_vec: np.ndarray,
    delta_L: np.ndarray,
    delta_W: np.ndarray
) -> np.ndarray:
    """
    Right-hand side of the ODE system for the 2D finite-volume method.
    Uses a generalized, gradient-based MUSCL formulation to maintain high-order 
    accuracy on both equidistant and non-equidistant grids.
    """
    n_L = len(L_vec)
    n_W = len(W_vec)
    
    N_temp = np.maximum(N_flat.reshape((n_L, n_W)), 0.0)
    
    # 2D Area Matrix
    cell_area = delta_L[:, None] * delta_W[None, :]
    f_temp = N_temp / cell_area
    epsilon = 1e-10

    # -------------------------------------------------------------------------
    # Fluxes in L direction
    # -------------------------------------------------------------------------
    f_pad_L = np.zeros((n_L + 2, n_W))
    f_pad_L[1:-1, :] = f_temp

    # Compute distances between adjacent cell centers (padded for boundaries)
    L_pad = np.concatenate([[L_vec[0] - delta_L[0]], L_vec, [L_vec[-1] + delta_L[-1]]])
    dist_L = L_pad[1:] - L_pad[:-1]

    # Calculate spatial gradients instead of simple differences
    grad_L = (f_pad_L[1:, :] - f_pad_L[:-1, :]) / dist_L[:, None]
    
    num_L = grad_L[1:, :] + epsilon
    den_L = grad_L[:-1, :] + epsilon
    r_L = num_L / den_L
    PHI_L = (np.abs(r_L) + r_L) / (1.0 + np.abs(r_L))

    f_borders_L = np.zeros((n_L + 1, n_W))
    # State interpolation to cell faces (x_{i+1/2})
    f_borders_L[1:, :] = f_temp + 0.5 * delta_L[:, None] * PHI_L * den_L
    f_borders_L = np.maximum(f_borders_L, 0.0)

    L_faces = L_vec + 0.5 * delta_L
    G_faces_L = growth_constant * (1.0 + gamma * L_faces)**p
    G_pos_L = np.zeros(n_L + 1)
    G_pos_L[1:] = G_faces_L

    J_L = G_pos_L[:, None] * f_borders_L
    dNdt_L = (J_L[:-1, :] - J_L[1:, :]) * delta_W[None, :]

    # -------------------------------------------------------------------------
    # Fluxes in W direction
    # -------------------------------------------------------------------------
    f_pad_W = np.zeros((n_L, n_W + 2))
    f_pad_W[:, 1:-1] = f_temp

    W_pad = np.concatenate([[W_vec[0] - delta_W[0]], W_vec, [W_vec[-1] + delta_W[-1]]])
    dist_W = W_pad[1:] - W_pad[:-1]

    grad_W = (f_pad_W[:, 1:] - f_pad_W[:, :-1]) / dist_W[None, :]
    
    num_W = grad_W[:, 1:] + epsilon
    den_W = grad_W[:, :-1] + epsilon
    r_W = num_W / den_W
    PHI_W = (np.abs(r_W) + r_W) / (1.0 + np.abs(r_W))

    f_borders_W = np.zeros((n_L, n_W + 1))
    f_borders_W[:, 1:] = f_temp + 0.5 * delta_W[None, :] * PHI_W * den_W
    f_borders_W = np.maximum(f_borders_W, 0.0)

    W_faces = W_vec + 0.5 * delta_W
    G_faces_W = growth_constant * (1.0 + gamma * W_faces)**p
    G_pos_W = np.zeros(n_W + 1)
    G_pos_W[1:] = G_faces_W

    J_W = G_pos_W[None, :] * f_borders_W
    dNdt_W = (J_W[:, :-1] - J_W[:, 1:]) * delta_L[:, None]

    dNdt = dNdt_L + dNdt_W
    return dNdt.flatten()

def compute_effective_areas(L: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute the effective 2D class areas from the current characteristic locations.
    """
    area_L = np.zeros_like(L)
    area_L[1:-1, :] = 0.5 * (L[2:, :] - L[:-2, :])
    area_L[0, :] = L[1, :] - L[0, :]
    area_L[-1, :] = L[-1, :] - L[-2, :]

    area_W = np.zeros_like(W)
    area_W[:, 1:-1] = 0.5 * (W[:, 2:] - W[:, :-2])
    area_W[:, 0] = W[:, 1] - W[:, 0]
    area_W[:, -1] = W[:, -1] - W[:, -2]

    return area_L * area_W

def analytical_solution_p1_2d(
    t: np.ndarray,
    L0: np.ndarray,
    W0: np.ndarray,
    f0: np.ndarray,
    growth_constant: float,
    gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the exact analytical 2D solution for the specific case of p = 1.
    
    This closed-form solution exists because the size-dependent growth kinetics
    in the length and width dimensions are structurally decoupled. For the linear
    growth laws G_L = k_g * (1 + gamma * L) and G_W = k_g * (1 + gamma * W),
    the two-dimensional population balance equation can be solved exactly along
    its characteristic trajectories.

    The characteristic curves for the spatial coordinates evolve as:
        L(t) = ((1 + gamma * L0) * exp(gamma * k_g * t) - 1) / gamma
        W(t) = ((1 + gamma * W0) * exp(gamma * k_g * t) - 1) / gamma

    The number density f(L, W, t) along these trajectories changes according to
    the divergence of the growth velocity field. Because the partial derivatives
    are constant (dG_L/dL = k_g * gamma and dG_W/dW = k_g * gamma), the material
    derivative of the density simplifies to:
        df/dt = -f * (dG_L/dL + dG_W/dW) = -2 * k_g * gamma * f

    Integrating this equation yields the exact exponential decay of the local
    number density, compensating for the continuous expansion of the phase space:
        f(t) = f0 * exp(-2 * k_g * gamma * t)
    """
    t = np.asarray(t)
    exp_forward = np.exp(gamma * growth_constant * t)[:, None, None]
    exp_decay = np.exp(-2.0 * gamma * growth_constant * t)[:, None, None]

    L_exact = ((1.0 + gamma * L0[None, :, :]) * exp_forward - 1.0) / gamma
    W_exact = ((1.0 + gamma * W0[None, :, :]) * exp_forward - 1.0) / gamma
    f_exact = f0[None, :, :] * exp_decay

    return L_exact, W_exact, f_exact

def simulate_moch(params: ModelParameters):
    LL_initial, WW_initial = create_initial_grid(params)
    f_initial = create_initial_number_density(LL_initial, WW_initial, params)
    y0 = build_initial_state(LL_initial, WW_initial, f_initial)

    n_total = LL_initial.size
    shape_2d = LL_initial.shape

    t_eval = np.linspace(0.0, params.final_time_min * 60.0, params.n_time_points)
    t_span = (t_eval[0], t_eval[-1])

    solution = solve_ivp(
        fun=growth_model_rhs,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(params.growth_constant, params.gamma, params.p, n_total),
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    Y = solution.y.T
    T = solution.t

    f_history = Y[:, :n_total].reshape((len(T), *shape_2d))
    L_history = Y[:, n_total:2*n_total].reshape((len(T), *shape_2d))
    W_history = Y[:, 2*n_total:].reshape((len(T), *shape_2d))

    total_number_history = np.zeros_like(T)
    for i in range(len(T)):
        areas = compute_effective_areas(L_history[i], W_history[i])
        total_number_history[i] = np.sum(f_history[i] * areas)

    return T, LL_initial, WW_initial, f_initial, L_history, W_history, f_history, total_number_history

def simulate_fv(params: ModelParameters, LL_grid: np.ndarray, WW_grid: np.ndarray, 
                delta_L: np.ndarray, delta_W: np.ndarray, f_initial: np.ndarray, T: np.ndarray):
    
    cell_area = delta_L[:, None] * delta_W[None, :]
    N_initial = f_initial * cell_area
    t_span = (T[0], T[-1])
    
    L_vec = LL_grid[:, 0]
    W_vec = WW_grid[0, :]

    solution = solve_ivp(
        fun=fv_model_rhs,
        t_span=t_span,
        y0=N_initial.flatten(),
        t_eval=T,
        args=(params.growth_constant, params.gamma, params.p, L_vec, W_vec, delta_L, delta_W),
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(f"FV ODE solver failed: {solution.message}")

    N_history = solution.y.T.reshape((len(T), len(L_vec), len(W_vec)))
    f_history_fv = N_history / cell_area
    
    total_number_fv = np.sum(f_history_fv * cell_area, axis=(1, 2))
    
    return f_history_fv, total_number_fv

def apply_publication_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.30,
        "legend.frameon": False,
        "mathtext.default": "regular",
    })

def style_axes(ax):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_total_number(T, total_number_moch, total_number_fv, total_number_initial):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    ax.plot(T / 60.0, total_number_moch, marker="o", markersize=4, label="Computed total number (MOCH)")
    ax.plot(T / 60.0, total_number_fv, marker="s", markersize=4, label="Computed total number (Finite Volume)")
    ax.axhline(total_number_initial, linestyle="--", linewidth=1.6, label="Initial total number")

    min_particles = min(np.min(total_number_moch), np.min(total_number_fv))
    max_particles = max(np.max(total_number_moch), np.max(total_number_fv))
    
    y_margin = 1.0 if (max_particles - min_particles) < 1.0 else 0.05 * (max_particles - min_particles)
    
    ax.set_ylim(min_particles - y_margin, max_particles + y_margin)

    ax.set_title("Total particle number over time")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Total particle number")
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    
    return fig, ax

def plot_marginal_distributions(
    T: np.ndarray, 
    L_moch: np.ndarray, W_moch: np.ndarray, f_moch: np.ndarray, 
    LL_grid_fv: np.ndarray, WW_grid_fv: np.ndarray, f_fv: np.ndarray,
    delta_L_fv: np.ndarray, delta_W_fv: np.ndarray,
    L_exact=None, W_exact=None, f_exact=None
):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    i_final = -1
    
    # -------------------------------------------------------------------------
    # 1. Data preparation for L-marginal distribution (integration over W)
    # -------------------------------------------------------------------------
    L_init_1d = L_moch[0, :, 0]
    W_init_widths = compute_1d_effective_widths(W_moch[0, 0, :])
    f_L_init = np.sum(f_moch[0, :, :] * W_init_widths, axis=1)

    L_moch_final_1d = L_moch[i_final, :, 0]
    W_moch_final_widths = compute_1d_effective_widths(W_moch[i_final, 0, :])
    f_L_moch_final = np.sum(f_moch[i_final, :, :] * W_moch_final_widths, axis=1)

    L_fv_1d = LL_grid_fv[:, 0]
    f_L_fv_final = np.sum(f_fv[i_final, :, :] * delta_W_fv[None, :], axis=1)

    if f_exact is not None:
        L_ex_final_1d = L_exact[i_final, :, 0]
        W_ex_final_widths = compute_1d_effective_widths(W_exact[i_final, 0, :])
        f_L_ex_final = np.sum(f_exact[i_final, :, :] * W_ex_final_widths, axis=1)

    # -------------------------------------------------------------------------
    # 2. Data preparation for W-marginal distribution (integration over L)
    # -------------------------------------------------------------------------
    W_init_1d = W_moch[0, 0, :]
    L_init_widths = compute_1d_effective_widths(L_moch[0, :, 0])
    f_W_init = np.sum(f_moch[0, :, :] * L_init_widths[:, None], axis=0)

    W_moch_final_1d = W_moch[i_final, 0, :]
    L_moch_final_widths = compute_1d_effective_widths(L_moch[i_final, :, 0])
    f_W_moch_final = np.sum(f_moch[i_final, :, :] * L_moch_final_widths[:, None], axis=0)

    W_fv_1d = WW_grid_fv[0, :]
    f_W_fv_final = np.sum(f_fv[i_final, :, :] * delta_L_fv[:, None], axis=0)

    if f_exact is not None:
        W_ex_final_1d = W_exact[i_final, 0, :]
        L_ex_final_widths = compute_1d_effective_widths(L_exact[i_final, :, 0])
        f_W_ex_final = np.sum(f_exact[i_final, :, :] * L_ex_final_widths[:, None], axis=0)

    # -------------------------------------------------------------------------
    # 3. Plotting Subplot 1: Length dimension
    # -------------------------------------------------------------------------
    axes[0].plot(L_init_1d, f_L_init, color='gray', linestyle=':', linewidth=2.0, label='Initial state ($t=0$)')
    axes[0].plot(L_fv_1d, f_L_fv_final, color='tab:blue', linestyle='-', linewidth=2.5, alpha=0.8, label='Finite Volume')
    axes[0].plot(L_moch_final_1d, f_L_moch_final, marker='o', color='tab:orange', markersize=4, linestyle='none', label='MOCH')
    
    if f_exact is not None:
        axes[0].plot(L_ex_final_1d, f_L_ex_final, color='black', linestyle='--', linewidth=1.5, label='Analytical')

    axes[0].set_title(f"Marginal Distribution along Length at $t = {T[-1]/60:.1f}$ min")
    axes[0].set_xlabel("Length, $L$ [$\\mu$m]")
    axes[0].set_ylabel("Marginal density, $f_L(L, t)$")
    axes[0].set_xlim([np.min(L_init_1d) - 10, np.max(L_moch_final_1d) + 20])
    style_axes(axes[0])
    axes[0].legend(loc='upper right')

    # -------------------------------------------------------------------------
    # 4. Plotting Subplot 2: Width dimension
    # -------------------------------------------------------------------------
    axes[1].plot(W_init_1d, f_W_init, color='gray', linestyle=':', linewidth=2.0, label='Initial state ($t=0$)')
    axes[1].plot(W_fv_1d, f_W_fv_final, color='tab:blue', linestyle='-', linewidth=2.5, alpha=0.8, label='Finite Volume')
    axes[1].plot(W_moch_final_1d, f_W_moch_final, marker='o', color='tab:orange', markersize=4, linestyle='none', label='MOCH')
    
    if f_exact is not None:
        axes[1].plot(W_ex_final_1d, f_W_ex_final, color='black', linestyle='--', linewidth=1.5, label='Analytical')

    axes[1].set_title(f"Marginal Distribution along Width at $t = {T[-1]/60:.1f}$ min")
    axes[1].set_xlabel("Width, $W$ [$\\mu$m]")
    axes[1].set_ylabel("Marginal density, $f_W(W, t)$")
    axes[1].set_xlim([np.min(W_init_1d) - 10, np.max(W_moch_final_1d) + 20])
    style_axes(axes[1])
    
    fig.tight_layout()
    return fig

def plot_2d_surface_evolution(L_history, W_history, f_history, title_prefix):
    fig = plt.figure(figsize=(12.0, 5.5))
    elevation_angle = 25
    azimuth_angle = -50

    # ---------------------------------------------------------
    # Subplot 1: Initial state
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(
        L_history[0], W_history[0], f_history[0],
        cmap='viridis', edgecolor='none', antialiased=True, alpha=0.9
    )
    
    ax1.set_title(f"{title_prefix} (Start)", pad=15)
    ax1.set_xlabel("Length, $L$ [$\\mu$m]", labelpad=10)
    ax1.set_ylabel("Width, $W$ [$\\mu$m]", labelpad=10)
    ax1.set_zlabel("Density, $f$", labelpad=10)
    ax1.set_xlim([0, 500])
    ax1.set_ylim([0, 500])
    ax1.view_init(elev=elevation_angle, azim=azimuth_angle)

    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax1.xaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})
    ax1.yaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})
    ax1.zaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})

    cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=15, pad=0.1)
    cb1.outline.set_visible(False)

    # ---------------------------------------------------------
    # Subplot 2: Final state
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(
        L_history[-1], W_history[-1], f_history[-1],
        cmap='viridis', edgecolor='none', antialiased=True, alpha=0.9
    )
    
    ax2.set_title(f"{title_prefix} (End)", pad=15)
    ax2.set_xlabel("Length, $L$ [$\\mu$m]", labelpad=10)
    ax2.set_ylabel("Width, $W$ [$\\mu$m]", labelpad=10)
    ax2.set_zlabel("Density, $f$", labelpad=10)
    ax2.set_xlim([0, 500])
    ax2.set_ylim([0, 500])
    ax2.view_init(elev=elevation_angle, azim=azimuth_angle)

    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax2.xaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})
    ax2.yaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})
    ax2.zaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})

    cb2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=15, pad=0.1)
    cb2.outline.set_visible(False)

    fig.tight_layout()
    return fig

def main():
    apply_publication_style()
    params = ModelParameters()

    # 1. Simulate MOCH
    T, LL_initial, WW_initial, f_initial, L_history_moch, W_history_moch, f_history_moch, total_number_moch = simulate_moch(params)
    total_number_initial = total_number_moch[0]

    # 2. Simulate Finite Volume
    LL_grid_fv, WW_grid_fv, delta_L_fv, delta_W_fv, f_initial_fv = create_extended_fv_grid(params, max_L=600.0, max_W=600.0)
    f_history_fv, total_number_fv = simulate_fv(params, LL_grid_fv, WW_grid_fv, delta_L_fv, delta_W_fv, f_initial_fv, T)

    # 3. Compute Analytical Solution
    if np.isclose(params.p, 1.0):
        L_exact, W_exact, f_exact = analytical_solution_p1_2d(
            t=T, L0=LL_initial, W0=WW_initial, f0=f_initial,
            growth_constant=params.growth_constant, gamma=params.gamma
        )
        
        max_abs_error_L = np.max(np.abs(L_history_moch[-1] - L_exact[-1]))
        max_abs_error_W = np.max(np.abs(W_history_moch[-1] - W_exact[-1]))
        max_abs_error_f = np.max(np.abs(f_history_moch[-1] - f_exact[-1]))

        print(f"Initial total particle number: {total_number_initial:.6f}")
        print(f"Final total particle number (MOCH): {total_number_moch[-1]:.6f}")
        print(f"Final total particle number (FV):   {total_number_fv[-1]:.6f}")
        print(f"Max abs error in L at final time (MOCH): {max_abs_error_L:.6e}")
        print(f"Max abs error in W at final time (MOCH): {max_abs_error_W:.6e}")
        print(f"Max abs error in f at final time (MOCH): {max_abs_error_f:.6e}")
    else:
        L_exact, W_exact, f_exact = None, None, None
        print(f"Initial total particle number: {total_number_initial:.6f}")
        print(f"Final total particle number (MOCH): {total_number_moch[-1]:.6f}")
        print(f"Final total particle number (FV):   {total_number_fv[-1]:.6f}")

    # 4. Visualizations
    plot_total_number(T, total_number_moch, total_number_fv, total_number_initial)
    plot_2d_surface_evolution(L_history_moch, W_history_moch, f_history_moch, "Density distribution MOCH")
    
    plot_marginal_distributions(
        T, L_history_moch, W_history_moch, f_history_moch, 
        LL_grid_fv, WW_grid_fv, f_history_fv, 
        delta_L_fv, delta_W_fv,
        L_exact, W_exact, f_exact
    )
    
    plt.show()

if __name__ == "__main__":
    main()