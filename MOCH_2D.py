import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp
from scipy.stats import norm

# =============================================================================
# 2D Population balance model:
# comparison between
# 1) the Method of Characteristics (MOCH)
# 2) a finite-volume (FV) discretization on a fixed size grid
#
# Governing equation
# ------------------
#     ∂f/∂t + ∂(G(L) f)/∂L + ∂(G(W) f)/∂W = 0
#
# with growth laws
#
#     G(L) = k * (1 + gamma * L)^p
#     G(W) = k * (1 + gamma * W)^p
#
# The two numerical formulations are kept explicitly separate throughout the
# code to avoid ambiguity in parameter usage, state definitions, and grid
# handling.
# =============================================================================


# =============================================================================
# Parameter containers and result containers
# =============================================================================

@dataclass(frozen=True)
class ModelParameters:
    """
    Immutable container for physical parameters, initialization settings,
    and solver-specific discretization parameters for the 2D case.
    """
    # -------------------------------------------------------------------------
    # Shared physical model parameters
    # -------------------------------------------------------------------------
    growth_constant: float = 0.02
    gamma: float = 0.005
    p: float = 1.0

    # -------------------------------------------------------------------------
    # Initial particle size distribution (2D Gaussian)
    # -------------------------------------------------------------------------
    mean_length: float = 90.0
    std_dev_length: float = 8.0
    
    mean_width: float = 150.0
    std_dev_width: float = 8.0

    total_number: float = 5000.0

    # -------------------------------------------------------------------------
    # Time discretization and reporting
    # -------------------------------------------------------------------------
    final_time_min: float = 100.0
    n_time_points: int = 10

    # -------------------------------------------------------------------------
    # MOCH-specific discretization
    # -------------------------------------------------------------------------
    moch_class_width_l: float = 1.5
    moch_class_width_w: float = 1.5

    # -------------------------------------------------------------------------
    # FV-specific discretization
    # -------------------------------------------------------------------------
    fv_min_length: float = 1.0
    fv_max_length: float = 500.0
    fv_num_classes_l: int = 250

    fv_min_width: float = 1.0
    fv_max_width: float = 500.0
    fv_num_classes_w: int = 250

    fv_discretization_method: str = "E"


@dataclass(frozen=True)
class MOCHSimulationResult:
    """
    Structured output of the 2D MOCH simulation.
    """
    time_s: np.ndarray
    moch_initial_grid_l: np.ndarray
    moch_initial_grid_w: np.ndarray
    moch_initial_density: np.ndarray
    moch_length_history: np.ndarray
    moch_width_history: np.ndarray
    moch_density_history: np.ndarray
    moch_total_number_history: np.ndarray


@dataclass(frozen=True)
class FVSimulationResult:
    """
    Structured output of the 2D FV simulation.
    """
    time_s: np.ndarray
    fv_class_edges_l: np.ndarray
    fv_class_edges_w: np.ndarray
    fv_pivot_points_l: np.ndarray
    fv_pivot_points_w: np.ndarray
    fv_class_widths_l: np.ndarray
    fv_class_widths_w: np.ndarray
    fv_initial_density: np.ndarray
    fv_inventory_history: np.ndarray
    fv_density_history: np.ndarray
    fv_total_number_history: np.ndarray


# =============================================================================
# Shared model utilities
# =============================================================================

def compute_growth_rate(size: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Evaluate the deterministic growth law G(x).
    """
    return params.growth_constant * (1.0 + params.gamma * size) ** params.p


def compute_growth_rate_derivative(size: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Evaluate the size derivative dG/dx of the growth law.
    """
    return (
        params.growth_constant
        * params.gamma
        * params.p
        * (1.0 + params.gamma * size) ** (params.p - 1.0)
    )


def create_time_grid_seconds(params: ModelParameters) -> np.ndarray:
    """
    Construct the common time grid used for MOCH and FV.
    """
    return np.linspace(0.0, params.final_time_min * 60.0, params.n_time_points)


def compute_nodal_control_widths(size_grid: np.ndarray) -> np.ndarray:
    """
    Compute effective nodal control widths for a one-dimensional monotone grid.
    """
    size_grid = np.asarray(size_grid)
    if size_grid.ndim != 1 or size_grid.size < 2:
        raise ValueError("Grid must be 1D and contain at least two points.")

    control_widths = np.empty_like(size_grid)
    control_widths[1:-1] = 0.5 * (size_grid[2:] - size_grid[:-2])
    control_widths[0] = size_grid[1] - size_grid[0]
    control_widths[-1] = size_grid[-1] - size_grid[-2]

    return control_widths


def evaluate_initial_number_density_2d(
    grid_l: np.ndarray,
    grid_w: np.ndarray,
    weights_l: np.ndarray,
    weights_w: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """
    Evaluate the initial 2D Gaussian number density and normalize it based on area.
    """
    raw_density = (
        norm.pdf(grid_l, loc=params.mean_length, scale=params.std_dev_length)
        * norm.pdf(grid_w, loc=params.mean_width, scale=params.std_dev_width)
    )
    
    area_matrix = weights_l[:, None] * weights_w[None, :]
    normalization = np.sum(raw_density * area_matrix)

    if normalization <= 0.0:
        raise ValueError("Initial density normalization is non-positive.")

    initial_density = params.total_number * raw_density / normalization
    return initial_density


# =============================================================================
# MOCH-specific functions
# =============================================================================

def create_moch_initial_grid_2d(params: ModelParameters) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Construct the initial MOCH characteristic 2D grid.
    """
    span_l = 4.0 * params.std_dev_length
    grid_start_l = params.mean_length - span_l
    grid_end_l = params.mean_length + span_l
    moch_grid_l = np.arange(
        grid_start_l, grid_end_l + 0.5 * params.moch_class_width_l, params.moch_class_width_l
    )

    span_w = 4.0 * params.std_dev_width
    grid_start_w = params.mean_width - span_w
    grid_end_w = params.mean_width + span_w
    moch_grid_w = np.arange(
        grid_start_w, grid_end_w + 0.5 * params.moch_class_width_w, params.moch_class_width_w
    )

    LL, WW = np.meshgrid(moch_grid_l, moch_grid_w, indexing='ij')
    shape_2d = LL.shape
    return LL.flatten(), WW.flatten(), shape_2d


def build_moch_initial_state_2d(
    moch_grid_l: np.ndarray,
    moch_grid_w: np.ndarray,
    moch_initial_density: np.ndarray,
) -> np.ndarray:
    """
    Assemble the MOCH ODE state vector for 2D.
    y = [L_1..L_N, W_1..W_N, f_1..f_N]
    """
    return np.concatenate([moch_grid_l, moch_grid_w, moch_initial_density.flatten()])


def moch_model_rhs_2d(
    t: float,
    state: np.ndarray,
    params: ModelParameters,
    n_moch_points: int,
) -> np.ndarray:
    """
    Right-hand side of the 2D MOCH ODE system.
    """
    del t

    length = state[:n_moch_points]
    width = state[n_moch_points:2 * n_moch_points]
    density = state[2 * n_moch_points:]

    dlength_dt = compute_growth_rate(length, params)
    dwidth_dt = compute_growth_rate(width, params)

    div_v = (
        compute_growth_rate_derivative(length, params) 
        + compute_growth_rate_derivative(width, params)
    )
    ddensity_dt = -density * div_v

    return np.concatenate([dlength_dt, dwidth_dt, ddensity_dt])


def solve_moch_simulation(
    params: ModelParameters,
    time_grid_s: np.ndarray,
) -> MOCHSimulationResult:
    """
    Run the 2D MOCH simulation.
    """
    moch_grid_l_flat, moch_grid_w_flat, shape_2d = create_moch_initial_grid_2d(params)
    
    # We need 1D grids for initial weight computation
    grid_l_1d = np.unique(moch_grid_l_flat)
    grid_w_1d = np.unique(moch_grid_w_flat)
    weights_l = compute_nodal_control_widths(grid_l_1d)
    weights_w = compute_nodal_control_widths(grid_w_1d)

    grid_l_2d = moch_grid_l_flat.reshape(shape_2d)
    grid_w_2d = moch_grid_w_flat.reshape(shape_2d)

    moch_initial_density = evaluate_initial_number_density_2d(
        grid_l=grid_l_2d,
        grid_w=grid_w_2d,
        weights_l=weights_l,
        weights_w=weights_w,
        params=params,
    )

    moch_initial_state = build_moch_initial_state_2d(
        moch_grid_l=moch_grid_l_flat,
        moch_grid_w=moch_grid_w_flat,
        moch_initial_density=moch_initial_density,
    )

    n_moch_points = moch_grid_l_flat.size

    moch_solution = solve_ivp(
        fun=moch_model_rhs_2d,
        t_span=(time_grid_s[0], time_grid_s[-1]),
        y0=moch_initial_state,
        t_eval=time_grid_s,
        args=(params, n_moch_points),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not moch_solution.success:
        raise RuntimeError(f"MOCH ODE solver failed: {moch_solution.message}")

    state_history = moch_solution.y.T
    moch_length_history = state_history[:, :n_moch_points].reshape((time_grid_s.size, *shape_2d))
    moch_width_history = state_history[:, n_moch_points:2 * n_moch_points].reshape((time_grid_s.size, *shape_2d))
    moch_density_history = state_history[:, 2 * n_moch_points:].reshape((time_grid_s.size, *shape_2d))

    moch_total_number_history = np.empty_like(time_grid_s)
    for i in range(time_grid_s.size):
        # Extract 1D marginal grids at time i assuming structured deformation
        current_l_1d = moch_length_history[i, :, 0]
        current_w_1d = moch_width_history[i, 0, :]
        current_weights_l = compute_nodal_control_widths(current_l_1d)
        current_weights_w = compute_nodal_control_widths(current_w_1d)
        
        area_matrix = current_weights_l[:, None] * current_weights_w[None, :]
        moch_total_number_history[i] = np.sum(moch_density_history[i] * area_matrix)

    return MOCHSimulationResult(
        time_s=moch_solution.t,
        moch_initial_grid_l=grid_l_2d,
        moch_initial_grid_w=grid_w_2d,
        moch_initial_density=moch_initial_density,
        moch_length_history=moch_length_history,
        moch_width_history=moch_width_history,
        moch_density_history=moch_density_history,
        moch_total_number_history=moch_total_number_history,
    )


def evaluate_analytical_solution_p1_2d(
    time_grid_s: np.ndarray,
    moch_initial_grid_l: np.ndarray,
    moch_initial_grid_w: np.ndarray,
    moch_initial_density: np.ndarray,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the analytical solution for the 2D special case p = 1.
    """
    exp_forward = np.exp(params.gamma * params.growth_constant * time_grid_s)[:, None, None]
    exp_decay = np.exp(-2.0 * params.gamma * params.growth_constant * time_grid_s)[:, None, None]

    exact_length_history = ((1.0 + params.gamma * moch_initial_grid_l[None, :, :]) * exp_forward - 1.0) / params.gamma
    exact_width_history = ((1.0 + params.gamma * moch_initial_grid_w[None, :, :]) * exp_forward - 1.0) / params.gamma
    exact_density_history = moch_initial_density[None, :, :] * exp_decay

    return exact_length_history, exact_width_history, exact_density_history


# =============================================================================
# FV-specific functions
# =============================================================================

def create_1d_grid(min_val: float, max_val: float, num_classes: int, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper to construct a 1D FV grid.
    """
    if max_val <= min_val:
        raise ValueError("max_val must be larger than min_val.")
    
    if method == "E":
        edges = np.linspace(min_val, max_val, num_classes + 1)
    elif method == "G1":
        growth_factor = (max_val / min_val) ** (1.0 / num_classes)
        indices = np.arange(1, num_classes + 1)
        edges = np.empty(num_classes + 1)
        edges[0] = min_val
        edges[1:] = min_val * growth_factor ** indices
    else:
        raise ValueError("Unsupported method.")

    pivot = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return edges, pivot, widths


def create_fv_grid_2d(params: ModelParameters) -> tuple[np.ndarray, ...]:
    """
    Construct the 2D finite-volume grid.
    """
    edges_l, pivot_l, widths_l = create_1d_grid(
        params.fv_min_length, params.fv_max_length, params.fv_num_classes_l, params.fv_discretization_method
    )
    edges_w, pivot_w, widths_w = create_1d_grid(
        params.fv_min_width, params.fv_max_width, params.fv_num_classes_w, params.fv_discretization_method
    )
    return edges_l, edges_w, pivot_l, pivot_w, widths_l, widths_w


def create_fv_initial_condition_2d(
    pivot_l: np.ndarray,
    pivot_w: np.ndarray,
    widths_l: np.ndarray,
    widths_w: np.ndarray,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the initial 2D FV number density and cell inventories.
    """
    LL, WW = np.meshgrid(pivot_l, pivot_w, indexing='ij')
    fv_initial_density = evaluate_initial_number_density_2d(
        grid_l=LL, grid_w=WW, weights_l=widths_l, weights_w=widths_w, params=params
    )
    area_matrix = widths_l[:, None] * widths_w[None, :]
    fv_initial_inventory = fv_initial_density * area_matrix

    return fv_initial_density, fv_initial_inventory


def fv_model_rhs_2d(
    t: float,
    fv_inventory_flat: np.ndarray,
    params: ModelParameters,
    fv_class_edges_l: np.ndarray,
    fv_class_edges_w: np.ndarray,
    fv_class_widths_l: np.ndarray,
    fv_class_widths_w: np.ndarray,
) -> np.ndarray:
    """
    Right-hand side of the 2D FV ODE system.
    Applies a flux limiter independently in the L and W directions.
    """
    del t

    n_l = fv_class_widths_l.size
    n_w = fv_class_widths_w.size
    epsilon = 1e-10

    fv_inventory = np.maximum(fv_inventory_flat.reshape((n_l, n_w)), 0.0)
    cell_areas = fv_class_widths_l[:, None] * fv_class_widths_w[None, :]
    fv_cell_density = fv_inventory / cell_areas

    # -------------------------------------------------------------------------
    # Flux reconstruction in Length direction
    # -------------------------------------------------------------------------
    f_pad_l = np.zeros((n_l + 2, n_w))
    f_pad_l[1:-1, :] = fv_cell_density

    num_l = f_pad_l[2:, :] - f_pad_l[1:-1, :] + epsilon
    den_l = f_pad_l[1:-1, :] - f_pad_l[:-2, :] + epsilon
    r_l = num_l / den_l
    phi_l = (np.abs(r_l) + r_l) / (1.0 + np.abs(r_l))

    f_face_l = np.zeros((n_l + 1, n_w))
    f_face_l[1:, :] = f_pad_l[1:-1, :] + 0.5 * phi_l * (f_pad_l[1:-1, :] - f_pad_l[:-2, :])
    f_face_l = np.maximum(f_face_l, 0.0)

    growth_rates_l = compute_growth_rate(fv_class_edges_l, params)
    flux_l = growth_rates_l[:, None] * f_face_l
    flux_l[0, :] = 0.0

    # -------------------------------------------------------------------------
    # Flux reconstruction in Width direction
    # -------------------------------------------------------------------------
    f_pad_w = np.zeros((n_l, n_w + 2))
    f_pad_w[:, 1:-1] = fv_cell_density

    num_w = f_pad_w[:, 2:] - f_pad_w[:, 1:-1] + epsilon
    den_w = f_pad_w[:, 1:-1] - f_pad_w[:, :-2] + epsilon
    r_w = num_w / den_w
    phi_w = (np.abs(r_w) + r_w) / (1.0 + np.abs(r_w))

    f_face_w = np.zeros((n_l, n_w + 1))
    f_face_w[:, 1:] = f_pad_w[:, 1:-1] + 0.5 * phi_w * (f_pad_w[:, 1:-1] - f_pad_w[:, :-2])
    f_face_w = np.maximum(f_face_w, 0.0)

    growth_rates_w = compute_growth_rate(fv_class_edges_w, params)
    flux_w = growth_rates_w[None, :] * f_face_w
    flux_w[:, 0] = 0.0

    # -------------------------------------------------------------------------
    # Update inventory
    # -------------------------------------------------------------------------
    d_inv_dt_l = (flux_l[:-1, :] - flux_l[1:, :]) * fv_class_widths_w[None, :]
    d_inv_dt_w = (flux_w[:, :-1] - flux_w[:, 1:]) * fv_class_widths_l[:, None]

    d_fv_inventory_dt = d_inv_dt_l + d_inv_dt_w
    return d_fv_inventory_dt.flatten()


def solve_fv_simulation(
    params: ModelParameters,
    time_grid_s: np.ndarray,
) -> FVSimulationResult:
    """
    Run the 2D FV simulation.
    """
    edges_l, edges_w, pivot_l, pivot_w, widths_l, widths_w = create_fv_grid_2d(params)

    fv_initial_density, fv_initial_inventory = create_fv_initial_condition_2d(
        pivot_l=pivot_l, pivot_w=pivot_w, widths_l=widths_l, widths_w=widths_w, params=params
    )

    fv_solution = solve_ivp(
        fun=fv_model_rhs_2d,
        t_span=(time_grid_s[0], time_grid_s[-1]),
        y0=fv_initial_inventory.flatten(),
        t_eval=time_grid_s,
        args=(params, edges_l, edges_w, widths_l, widths_w),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not fv_solution.success:
        raise RuntimeError(f"FV ODE solver failed: {fv_solution.message}")

    n_l = widths_l.size
    n_w = widths_w.size
    cell_areas = widths_l[:, None] * widths_w[None, :]

    fv_inventory_history = fv_solution.y.T.reshape((time_grid_s.size, n_l, n_w))
    fv_density_history = fv_inventory_history / cell_areas[None, :, :]
    fv_total_number_history = np.sum(fv_inventory_history, axis=(1, 2))

    return FVSimulationResult(
        time_s=fv_solution.t,
        fv_class_edges_l=edges_l,
        fv_class_edges_w=edges_w,
        fv_pivot_points_l=pivot_l,
        fv_pivot_points_w=pivot_w,
        fv_class_widths_l=widths_l,
        fv_class_widths_w=widths_w,
        fv_initial_density=fv_initial_density,
        fv_inventory_history=fv_inventory_history,
        fv_density_history=fv_density_history,
        fv_total_number_history=fv_total_number_history,
    )


# =============================================================================
# Plotting utilities
# =============================================================================

def apply_publication_plot_style() -> None:
    """
    Apply a clean plotting style suitable for publication-oriented figures.
    """
    plt.rcParams.update(
        {
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
        }
    )


def style_publication_axes(ax) -> None:
    """
    Apply a minimal axis style that emphasizes the data.
    """
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_total_particle_number_comparison(
    time_s: np.ndarray,
    moch_total_number_history: np.ndarray,
    fv_total_number_history: np.ndarray,
    total_number_initial: float,
):
    """
    Plot the total particle number predicted by MOCH and FV.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    ax.plot(time_s / 60.0, moch_total_number_history, marker="o", markersize=4, label="MOCH")
    ax.plot(time_s / 60.0, fv_total_number_history, marker="s", markersize=4, label="Finite volume")
    ax.axhline(total_number_initial, linestyle="--", linewidth=1.6, label="Initial total number")

    y_min = min(np.min(moch_total_number_history), np.min(fv_total_number_history))
    y_max = max(np.max(moch_total_number_history), np.max(fv_total_number_history))
    ax.set_ylim(np.floor(y_min) - 1.0, np.ceil(y_max) + 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    ax.set_title("Total particle number over time")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Total particle number")
    style_publication_axes(ax)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_3d_combined_evolution(
    result,
    method_name: str,
    params: ModelParameters
):
    """
    Plot the initial and final 3D PSD surface combined in a single figure for direct comparison.
    Works for both MOCH and FV results.
    """
    fig = plt.figure(figsize=(10.0, 6.0))
    ax = fig.add_subplot(111, projection='3d')
    elevation_angle = 25
    azimuth_angle = -50

    # Determine data sources based on the result class
    if isinstance(result, MOCHSimulationResult):
        L_hist = result.moch_length_history
        W_hist = result.moch_width_history
        f_hist = result.moch_density_history
        
    elif isinstance(result, FVSimulationResult):
        # For FV, the results are on a fixed grid
        L_vec = result.fv_pivot_points_l
        W_vec = result.fv_pivot_points_w
        f_hist = result.fv_density_history
        L_grid_fv, W_grid_fv = np.meshgrid(L_vec, W_vec, indexing='ij')
        
        # Tile the stationary grids into historical arrays
        L_hist = np.tile(L_grid_fv[None, :, :], (result.time_s.size, 1, 1))
        W_hist = np.tile(W_grid_fv[None, :, :], (result.time_s.size, 1, 1))

    else:
        raise ValueError("Unsupported result type.")

    # Plot Start (time index 0)
    surf1 = ax.plot_surface(
        L_hist[0], W_hist[0], f_hist[0],
        cmap='viridis', edgecolor='none', antialiased=True, alpha=0.9
    )
    
    # Plot End (time index -1)
    surf2 = ax.plot_surface(
        L_hist[-1], W_hist[-1], f_hist[-1],
        cmap='viridis', edgecolor='none', antialiased=True, alpha=0.9
    )

    # Apply global axes limits to accommodate both distributions and full peak height
    ax.set_xlim([params.fv_min_length - 10.0, params.fv_max_length + 10.0])
    ax.set_ylim([params.fv_min_width - 10.0, params.fv_max_width + 10.0])
    # Dynamic Z limits to fit the full peak height with some headroom
    ax.set_zlim([0, np.max(f_hist) * 1.1]) 

    # Minimal axes style with a white pane background for clarity
    ax.set_title(f"Combined Evolution of the Particle Size Distribution ({method_name})", pad=15)
    ax.set_xlabel("Length, $L$ [$\\mu$m]", labelpad=10)
    ax.set_ylabel("Width, $W$ [$\\mu$m]", labelpad=10)
    ax.set_zlabel("Density, $f$", labelpad=10)
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    ax.xaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})
    ax.yaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})
    ax.zaxis._axinfo["grid"].update({"color": "gray", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5})

    # Add 3D text annotations pointing to the peaks
    # For start peak
    max_idx_start = np.unravel_index(np.argmax(f_hist[0]), f_hist[0].shape)
    peak_l_start = L_hist[0][max_idx_start]
    peak_w_start = W_hist[0][max_idx_start]
    peak_f_start = f_hist[0][max_idx_start]
    ax.text(peak_l_start, peak_w_start, peak_f_start * 1.05, "Start (t=0)", color='black', fontsize=10, fontweight='bold', ha='center', va='bottom')

    # For end peak
    max_idx_end = np.unravel_index(np.argmax(f_hist[-1]), f_hist[-1].shape)
    peak_l_end = L_hist[-1][max_idx_end]
    peak_w_end = W_hist[-1][max_idx_end]
    peak_f_end = f_hist[-1][max_idx_end]
    ax.text(peak_l_end, peak_w_end, peak_f_end * 1.05, f"End (t={result.time_s[-1]/60.0:.1f} min)", color='black', fontsize=10, fontweight='bold', ha='center', va='bottom')

    # Add a single colorbar for density across the entire combined Z-range
    cb = fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cb.outline.set_visible(False)
    cb.set_label("Density, f")

    fig.tight_layout()
    return fig


def plot_marginal_distributions(
    moch_result: MOCHSimulationResult,
    fv_result: FVSimulationResult,
    exact_density_history: np.ndarray | None = None,
):
    """
    Plot the 1D marginal distributions along Length and Width at the final time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    i_final = -1
    final_time_min = moch_result.time_s[i_final] / 60.0

    # MOCH Marginals
    L_init_1d = moch_result.moch_length_history[0, :, 0]
    W_init_widths = compute_nodal_control_widths(moch_result.moch_width_history[0, 0, :])
    f_L_init = np.sum(moch_result.moch_density_history[0, :, :] * W_init_widths, axis=1)

    L_moch_final_1d = moch_result.moch_length_history[i_final, :, 0]
    W_moch_final_widths = compute_nodal_control_widths(moch_result.moch_width_history[i_final, 0, :])
    f_L_moch_final = np.sum(moch_result.moch_density_history[i_final, :, :] * W_moch_final_widths, axis=1)

    W_init_1d = moch_result.moch_width_history[0, 0, :]
    L_init_widths = compute_nodal_control_widths(moch_result.moch_length_history[0, :, 0])
    f_W_init = np.sum(moch_result.moch_density_history[0, :, :] * L_init_widths[:, None], axis=0)

    W_moch_final_1d = moch_result.moch_width_history[i_final, 0, :]
    L_moch_final_widths = compute_nodal_control_widths(moch_result.moch_length_history[i_final, :, 0])
    f_W_moch_final = np.sum(moch_result.moch_density_history[i_final, :, :] * L_moch_final_widths[:, None], axis=0)

    # FV Marginals
    f_L_fv_final = np.sum(fv_result.fv_density_history[i_final, :, :] * fv_result.fv_class_widths_w[None, :], axis=1)
    f_W_fv_final = np.sum(fv_result.fv_density_history[i_final, :, :] * fv_result.fv_class_widths_l[:, None], axis=0)

    # Analytical Marginals (if available, assume p=1 special case)
    if exact_density_history is not None:
        L_ex_final_1d = moch_result.moch_length_history[i_final, :, 0] # Trajectories match numerical MOCH for p=1
        W_ex_final_widths = compute_nodal_control_widths(moch_result.moch_width_history[i_final, 0, :])
        f_L_ex_final = np.sum(exact_density_history[i_final, :, :] * W_ex_final_widths, axis=1)

        W_ex_final_1d = moch_result.moch_width_history[i_final, 0, :]
        L_ex_final_widths = compute_nodal_control_widths(moch_result.moch_length_history[i_final, :, 0])
        f_W_ex_final = np.sum(exact_density_history[i_final, :, :] * L_ex_final_widths[:, None], axis=0)

    # Plot marginal PSD along Length
    axes[0].plot(L_init_1d, f_L_init, color='gray', linestyle=':', label='Initial state ($t=0$)')
    axes[0].plot(fv_result.fv_pivot_points_l, f_L_fv_final, color='tab:blue', linewidth=2.5, alpha=0.8, label='Finite Volume')
    axes[0].plot(L_moch_final_1d, f_L_moch_final, marker='o', color='tab:orange', markersize=4, linestyle='none', label='MOCH')
    if exact_density_history is not None:
        axes[0].plot(L_ex_final_1d, f_L_ex_final, color='black', linestyle='--', linewidth=1.5, label='Analytical')

    axes[0].set_title(f"Marginal Distribution (Length) at $t = {final_time_min:.1f}$ min")
    axes[0].set_xlabel("Length, $L$ [$\\mu$m]")
    axes[0].set_ylabel("Marginal density, $f_L(L, t)$")
    style_publication_axes(axes[0])
    axes[0].legend()

    # Plot marginal PSD along Width
    axes[1].plot(W_init_1d, f_W_init, color='gray', linestyle=':', label='Initial state ($t=0$)')
    axes[1].plot(fv_result.fv_pivot_points_w, f_W_fv_final, color='tab:blue', linewidth=2.5, alpha=0.8, label='Finite Volume')
    axes[1].plot(W_moch_final_1d, f_W_moch_final, marker='o', color='tab:orange', markersize=4, linestyle='none', label='MOCH')
    if exact_density_history is not None:
        axes[1].plot(W_ex_final_1d, f_W_ex_final, color='black', linestyle='--', linewidth=1.5, label='Analytical')

    axes[1].set_title(f"Marginal Distribution (Width) at $t = {final_time_min:.1f}$ min")
    axes[1].set_xlabel("Width, $W$ [$\\mu$m]")
    axes[1].set_ylabel("Marginal density, $f_W(W, t)$")
    style_publication_axes(axes[1])
    axes[1].legend()

    fig.tight_layout()
    return fig


# =============================================================================
# Reporting
# =============================================================================

def print_simulation_summary(
    params: ModelParameters,
    moch_result: MOCHSimulationResult,
    fv_result: FVSimulationResult,
    exact_density_history: np.ndarray | None = None,
) -> None:
    """
    Print a compact simulation summary.
    """
    print(f"Initial total particle number: {params.total_number:.6f}")
    print(f"Final total particle number (MOCH): {moch_result.moch_total_number_history[-1]:.6f}")
    print(f"Final total particle number (FV):   {fv_result.fv_total_number_history[-1]:.6f}")

    if exact_density_history is not None:
        max_abs_error_density = np.max(
            np.abs(moch_result.moch_density_history[-1] - exact_density_history[-1])
        )
        print(f"Maximum absolute error in f at final time (MOCH): {max_abs_error_density:.6e}")


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    """
    Execute the 2D MOCH-versus-FV comparison workflow.
    """
    apply_publication_plot_style()
    params = ModelParameters()

    # Create a common reporting grid for exact time comparison
    time_grid_s = create_time_grid_seconds(params)

    # Solve both numerical formulations
    moch_result = solve_moch_simulation(params, time_grid_s)
    fv_result = solve_fv_simulation(params, time_grid_s)

    # Evaluate analytical solution if applicable (e.g., p=1 special case)
    if np.isclose(params.p, 1.0):
        exact_length_history, exact_width_history, exact_density_history = evaluate_analytical_solution_p1_2d(
            time_grid_s=moch_result.time_s,
            moch_initial_grid_l=moch_result.moch_initial_grid_l,
            moch_initial_grid_w=moch_result.moch_initial_grid_w,
            moch_initial_density=moch_result.moch_initial_density,
            params=params,
        )
    else:
        exact_density_history = None

    print_simulation_summary(
        params=params,
        moch_result=moch_result,
        fv_result=fv_result,
        exact_density_history=exact_density_history,
    )

    # Visualizations
    plot_total_particle_number_comparison(
        time_s=moch_result.time_s,
        moch_total_number_history=moch_result.moch_total_number_history,
        fv_total_number_history=fv_result.fv_total_number_history,
        total_number_initial=params.total_number,
    )
    
    # Combined 3D plot for direct MOCH Start/End comparison
    plot_3d_combined_evolution(moch_result, "MOCH", params)
    
    # Combined 3D plot for direct FV Start/End comparison
    plot_3d_combined_evolution(fv_result, "FV", params)

    # Marginal distributions at final time for full comparison
    plot_marginal_distributions(moch_result, fv_result, exact_density_history)

    plt.show()

if __name__ == "__main__":
    main()