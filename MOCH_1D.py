import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp
from scipy.stats import norm


# =============================================================================
# Population balance model:
# comparison between
# 1) the Method of Characteristics (MOCH)
# 2) a finite-volume (FV) discretization on a fixed size grid
#
# Governing equation
# ------------------
#     ∂f/∂t + ∂(G(L) f)/∂L = 0
#
# with growth law
#
#     G(L) = k * (1 + gamma * L)^p
#
# The two numerical formulations are kept explicitly separate throughout the
# code to avoid ambiguity in parameter usage, state definitions, and grid
# handling. This is particularly important in a method-comparison study, where
# the physical model must remain identical while the numerical representation
# changes.
# =============================================================================


# =============================================================================
# Parameter containers and result containers
# =============================================================================

@dataclass(frozen=True)
class ModelParameters:
    """
    Immutable container for physical parameters, initialization settings,
    and solver-specific discretization parameters.

    Notes
    -----
    The parameter set is intentionally partitioned into
    1. model parameters shared by MOCH and FV,
    2. initial PSD parameters,
    3. time-reporting settings,
    4. MOCH-specific grid settings,
    5. FV-specific grid settings.

    This separation makes it immediately visible which quantities define the
    underlying population balance model and which only control a particular
    numerical discretization.

    Units
    -----
    growth_constant : µm/s
    mean_size       : µm
    std_dev         : µm
    moch_class_width: µm
    final_time_min  : min
    fv_min_size     : µm
    fv_max_size     : µm
    """
    # -------------------------------------------------------------------------
    # Shared physical model parameters
    # -------------------------------------------------------------------------
    growth_constant: float = 0.02
    # Lumped growth constant in the growth law [µm/s].

    gamma: float = 0.005
    # Coefficient governing size dependence in the growth law.

    p: float = 1.0
    # Exponent in the growth law.

    # -------------------------------------------------------------------------
    # Initial particle size distribution
    # -------------------------------------------------------------------------
    mean_size: float = 90.0
    # Mean of the initial Gaussian particle size distribution [µm].

    std_dev: float = 8.0
    # Standard deviation of the initial Gaussian particle size distribution [µm].

    total_number: float = 5000.0
    # Prescribed total initial particle number.

    # -------------------------------------------------------------------------
    # Time discretization and reporting
    # -------------------------------------------------------------------------
    final_time_min: float = 100.0
    # Final simulation time reported in minutes.

    n_time_points: int = 10
    # Number of time points stored for post-processing.

    # -------------------------------------------------------------------------
    # MOCH-specific discretization
    # -------------------------------------------------------------------------
    moch_class_width: float = 1.5
    # Initial spacing between MOCH characteristic points [µm].

    # -------------------------------------------------------------------------
    # FV-specific discretization
    # -------------------------------------------------------------------------
    fv_min_size: float = 1.0
    # Lower bound of the FV size domain [µm].
    # Must be strictly positive for geometric FV discretization ("G1").

    fv_max_size: float = 450.0
    # Upper bound of the FV size domain [µm].

    fv_num_classes: int = 300
    # Number of FV control volumes.

    fv_discretization_method: str = "E"
    # FV grid construction method:
    # "E"  : uniform spacing in size
    # "G1" : geometric spacing with finer resolution at small sizes
    # "G2" : cubic mapping, i.e. uniform spacing in x^(1/3)


@dataclass(frozen=True)
class MOCHSimulationResult:
    """
    Structured output of the MOCH simulation.
    """
    time_s: np.ndarray
    moch_initial_grid: np.ndarray
    moch_initial_density: np.ndarray
    moch_size_history: np.ndarray
    moch_density_history: np.ndarray
    moch_total_number_history: np.ndarray


@dataclass(frozen=True)
class FVSimulationResult:
    """
    Structured output of the FV simulation.
    """
    time_s: np.ndarray
    fv_class_edges: np.ndarray
    fv_pivot_points: np.ndarray
    fv_class_widths: np.ndarray
    fv_initial_density: np.ndarray
    fv_inventory_history: np.ndarray
    fv_density_history: np.ndarray
    fv_total_number_history: np.ndarray


# =============================================================================
# Shared model utilities
# =============================================================================

def compute_growth_rate(size: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Evaluate the deterministic growth law G(L).

    Parameters
    ----------
    size : np.ndarray
        Size coordinate(s) at which the growth rate is evaluated.
    params : ModelParameters
        Model parameter set.

    Returns
    -------
    np.ndarray
        Growth rate values G(L).
    """
    return params.growth_constant * (1.0 + params.gamma * size) ** params.p


def compute_growth_rate_derivative(size: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Evaluate the size derivative dG/dL of the growth law.

    This derivative enters the MOCH evolution equation for the number density
    along the moving characteristics.

    Parameters
    ----------
    size : np.ndarray
        Size coordinate(s) at which dG/dL is evaluated.
    params : ModelParameters
        Model parameter set.

    Returns
    -------
    np.ndarray
        Values of dG/dL.
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

    The governing equations are integrated in seconds because the growth
    constant is specified in µm/s.

    Parameters
    ----------
    params : ModelParameters
        Model parameter set.

    Returns
    -------
    np.ndarray
        One-dimensional time grid in seconds.
    """
    return np.linspace(0.0, params.final_time_min * 60.0, params.n_time_points)


def compute_nodal_control_widths(size_grid: np.ndarray) -> np.ndarray:
    """
    Compute effective nodal control widths for a one-dimensional monotone grid.

    The interior weight of node i is defined as half the distance to the left
    neighbor plus half the distance to the right neighbor. At the boundaries,
    the nearest-neighbor spacing is used.

    This construction is useful when a nodal number density is converted into a
    discrete approximation of the integral ∫ f(L) dL.

    Parameters
    ----------
    size_grid : np.ndarray
        Strictly increasing one-dimensional grid.

    Returns
    -------
    np.ndarray
        Effective control widths associated with the grid nodes.

    Raises
    ------
    ValueError
        If the grid is not one-dimensional, contains fewer than two points,
        or is not strictly increasing.
    """
    size_grid = np.asarray(size_grid)

    if size_grid.ndim != 1:
        raise ValueError("size_grid must be one-dimensional.")

    if size_grid.size < 2:
        raise ValueError("size_grid must contain at least two points.")

    if not np.all(np.diff(size_grid) > 0.0):
        raise ValueError("size_grid must be strictly increasing.")

    control_widths = np.empty_like(size_grid)
    control_widths[1:-1] = 0.5 * (size_grid[2:] - size_grid[:-2])
    control_widths[0] = size_grid[1] - size_grid[0]
    control_widths[-1] = size_grid[-1] - size_grid[-2]

    return control_widths


def evaluate_initial_number_density(
    size_grid: np.ndarray,
    integration_weights: np.ndarray,
    params: ModelParameters,
) -> np.ndarray:
    """
    Evaluate the initial Gaussian number density on a prescribed grid and scale
    it such that the discrete particle number matches the prescribed total.

    Parameters
    ----------
    size_grid : np.ndarray
        Grid points at which the initial density is sampled.
    integration_weights : np.ndarray
        Discrete integration weights associated with the sampling points.
    params : ModelParameters
        Model parameter set.

    Returns
    -------
    np.ndarray
        Initial number density values on the supplied grid.

    Raises
    ------
    ValueError
        If the discrete normalization factor is non-positive.
    """
    raw_density = norm.pdf(size_grid, loc=params.mean_size, scale=params.std_dev)
    normalization = np.sum(raw_density * integration_weights)

    if normalization <= 0.0:
        raise ValueError("Initial density normalization is non-positive.")

    initial_density = params.total_number * raw_density / normalization
    return initial_density


# =============================================================================
# MOCH-specific functions
# =============================================================================

def create_moch_initial_grid(params: ModelParameters) -> np.ndarray:
    """
    Construct the initial MOCH characteristic grid.

    The grid spans approximately ±4 standard deviations around the mean of the
    initial Gaussian PSD. This is usually sufficient to capture the relevant
    support of the initial distribution while keeping the number of tracked
    characteristics moderate.

    Parameters
    ----------
    params : ModelParameters
        Model parameter set.

    Returns
    -------
    np.ndarray
        Initial MOCH characteristic positions.
    """
    span = 4.0 * params.std_dev
    grid_start = params.mean_size - span
    grid_end = params.mean_size + span

    moch_initial_grid = np.arange(
        grid_start,
        grid_end + 0.5 * params.moch_class_width,
        params.moch_class_width,
    )
    return moch_initial_grid


def build_moch_initial_state(
    moch_initial_grid: np.ndarray,
    moch_initial_density: np.ndarray,
) -> np.ndarray:
    """
    Assemble the MOCH ODE state vector.

    State ordering
    --------------
    y = [L_1, ..., L_N, f_1, ..., f_N]

    Parameters
    ----------
    moch_initial_grid : np.ndarray
        Initial characteristic positions.
    moch_initial_density : np.ndarray
        Initial number density sampled at the characteristic positions.

    Returns
    -------
    np.ndarray
        Initial MOCH state vector.
    """
    return np.concatenate([moch_initial_grid, moch_initial_density])


def moch_model_rhs(
    t: float,
    state: np.ndarray,
    params: ModelParameters,
    n_moch_points: int,
) -> np.ndarray:
    """
    Right-hand side of the MOCH ODE system.

    The characteristic positions evolve according to

        dL/dt = G(L),

    whereas the number density along each characteristic satisfies

        df/dt = -f * dG/dL.

    The formulation corresponds to a pure-growth population balance without
    additional source terms such as nucleation, aggregation, or breakage.

    Parameters
    ----------
    t : float
        Time in seconds.
    state : np.ndarray
        Current MOCH state vector.
    params : ModelParameters
        Model parameter set.
    n_moch_points : int
        Number of characteristic points.

    Returns
    -------
    np.ndarray
        Time derivative of the MOCH state vector.
    """
    del t  # time does not enter the autonomous system explicitly

    size = state[:n_moch_points]
    density = state[n_moch_points:]

    dsize_dt = compute_growth_rate(size, params)
    ddensity_dt = -density * compute_growth_rate_derivative(size, params)

    return np.concatenate([dsize_dt, ddensity_dt])


def solve_moch_simulation(
    params: ModelParameters,
    time_grid_s: np.ndarray,
) -> MOCHSimulationResult:
    """
    Run the MOCH simulation on the prescribed time grid.

    Parameters
    ----------
    params : ModelParameters
        Model parameter set.
    time_grid_s : np.ndarray
        Output time grid in seconds.

    Returns
    -------
    MOCHSimulationResult
        Structured MOCH simulation output.
    """
    moch_initial_grid = create_moch_initial_grid(params)
    moch_initial_weights = compute_nodal_control_widths(moch_initial_grid)

    moch_initial_density = evaluate_initial_number_density(
        size_grid=moch_initial_grid,
        integration_weights=moch_initial_weights,
        params=params,
    )

    moch_initial_state = build_moch_initial_state(
        moch_initial_grid=moch_initial_grid,
        moch_initial_density=moch_initial_density,
    )

    n_moch_points = moch_initial_grid.size

    moch_solution = solve_ivp(
        fun=moch_model_rhs,
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
    moch_size_history = state_history[:, :n_moch_points]
    moch_density_history = state_history[:, n_moch_points:]

    moch_total_number_history = np.empty_like(moch_solution.t)
    for i in range(moch_solution.t.size):
        current_weights = compute_nodal_control_widths(moch_size_history[i, :])
        moch_total_number_history[i] = np.sum(
            moch_density_history[i, :] * current_weights
        )

    return MOCHSimulationResult(
        time_s=moch_solution.t,
        moch_initial_grid=moch_initial_grid,
        moch_initial_density=moch_initial_density,
        moch_size_history=moch_size_history,
        moch_density_history=moch_density_history,
        moch_total_number_history=moch_total_number_history,
    )


def evaluate_analytical_solution_p1(
    time_grid_s: np.ndarray,
    moch_initial_grid: np.ndarray,
    moch_initial_density: np.ndarray,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the analytical solution for the special case p = 1.

    For p = 1, the characteristic trajectories and the density decay are

        L(t) = ((1 + gamma * L0) * exp(gamma * k * t) - 1) / gamma
        f(t) = f0(L0) * exp(-gamma * k * t)

    Parameters
    ----------
    time_grid_s : np.ndarray
        Time points in seconds.
    moch_initial_grid : np.ndarray
        Initial characteristic positions.
    moch_initial_density : np.ndarray
        Initial density values on the MOCH grid.
    params : ModelParameters
        Model parameter set.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Exact size history and exact density history.
    """
    exp_forward = np.exp(params.gamma * params.growth_constant * time_grid_s)[:, None]
    exp_decay = np.exp(-params.gamma * params.growth_constant * time_grid_s)[:, None]

    exact_size_history = (
        (1.0 + params.gamma * moch_initial_grid[None, :]) * exp_forward - 1.0
    ) / params.gamma

    exact_density_history = moch_initial_density[None, :] * exp_decay

    return exact_size_history, exact_density_history


# =============================================================================
# FV-specific functions
# =============================================================================

def create_fv_grid(
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the finite-volume grid.

    Returns
    -------
    fv_class_edges : np.ndarray
        Coordinates of the FV cell boundaries.
    fv_pivot_points : np.ndarray
        Coordinates of the FV cell centers.
    fv_class_widths : np.ndarray
        Widths of the FV control volumes.

    Raises
    ------
    ValueError
        If the FV grid specification is inconsistent.
    """
    fv_min_size = params.fv_min_size
    fv_max_size = params.fv_max_size
    fv_num_classes = params.fv_num_classes
    fv_discretization_method = params.fv_discretization_method

    if fv_max_size <= fv_min_size:
        raise ValueError("fv_max_size must be larger than fv_min_size.")

    if fv_num_classes <= 0:
        raise ValueError("fv_num_classes must be a positive integer.")

    if fv_discretization_method == "E":
        # Uniform discretization in the physical size coordinate.
        fv_class_edges = np.linspace(
            fv_min_size,
            fv_max_size,
            fv_num_classes + 1,
        )

    elif fv_discretization_method == "G1":
        # Geometric spacing provides increased resolution at small sizes.
        if fv_min_size <= 0.0:
            raise ValueError(
                "fv_min_size must be strictly positive for 'G1' discretization."
            )

        fv_growth_factor = (fv_max_size / fv_min_size) ** (1.0 / fv_num_classes)
        fv_class_indices = np.arange(1, fv_num_classes + 1)

        fv_class_edges = np.empty(fv_num_classes + 1)
        fv_class_edges[0] = fv_min_size
        fv_class_edges[1:] = fv_min_size * fv_growth_factor ** fv_class_indices

    elif fv_discretization_method == "G2":
        # Uniform spacing in the transformed coordinate x^(1/3).
        fv_class_edges = np.linspace(
            fv_min_size ** (1.0 / 3.0),
            fv_max_size ** (1.0 / 3.0),
            fv_num_classes + 1,
        ) ** 3

    else:
        raise ValueError(
            f"Unsupported FV discretization method '{fv_discretization_method}'. "
            "Use 'E', 'G1', or 'G2'."
        )

    fv_pivot_points = 0.5 * (fv_class_edges[:-1] + fv_class_edges[1:])
    fv_class_widths = fv_class_edges[1:] - fv_class_edges[:-1]

    return fv_class_edges, fv_pivot_points, fv_class_widths


def create_fv_initial_condition(
    fv_pivot_points: np.ndarray,
    fv_class_widths: np.ndarray,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the initial FV number density and the corresponding cell
    inventories.

    The FV state variable is the cell inventory

        Q_i = ∫_{cell i} f(L) dL,

    which is approximated from the initial density sampled at the pivot points.

    Parameters
    ----------
    fv_pivot_points : np.ndarray
        FV cell centers.
    fv_class_widths : np.ndarray
        FV cell widths.
    params : ModelParameters
        Model parameter set.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Initial FV density and initial FV inventory.
    """
    fv_initial_density = evaluate_initial_number_density(
        size_grid=fv_pivot_points,
        integration_weights=fv_class_widths,
        params=params,
    )

    fv_initial_inventory = fv_initial_density * fv_class_widths
    return fv_initial_density, fv_initial_inventory


def fv_model_rhs(
    t: float,
    fv_inventory: np.ndarray,
    params: ModelParameters,
    fv_class_edges: np.ndarray,
    fv_class_widths: np.ndarray,
) -> np.ndarray:
    del t

    n_cells = fv_inventory.size
    epsilon = 1e-10

    fv_inventory = np.maximum(fv_inventory, 0.0)
    fv_cell_density = fv_inventory / fv_class_widths

    # Padded array for limiter-based face reconstruction
    fv_density_padded = np.zeros(n_cells + 2)
    fv_density_padded[1:-1] = fv_cell_density

    num = fv_density_padded[2:] - fv_density_padded[1:-1] + epsilon
    den = fv_density_padded[1:-1] - fv_density_padded[:-2] + epsilon
    r = num / den
    phi = (np.abs(r) + r) / (1.0 + np.abs(r))

    fv_face_density = np.zeros(n_cells + 1)
    fv_face_density[1:] = (
        fv_density_padded[1:-1]
        + 0.5 * phi * (fv_density_padded[1:-1] - fv_density_padded[:-2])
    )
    fv_face_density = np.maximum(fv_face_density, 0.0)

    fv_face_growth_rates = compute_growth_rate(fv_class_edges, params)
    fv_face_flux = fv_face_growth_rates * fv_face_density
    fv_face_flux[0] = 0.0

    d_fv_inventory_dt = fv_face_flux[:-1] - fv_face_flux[1:]
    return d_fv_inventory_dt


def solve_fv_simulation(
    params: ModelParameters,
    time_grid_s: np.ndarray,
) -> FVSimulationResult:
    """
    Run the FV simulation on the prescribed time grid.

    Parameters
    ----------
    params : ModelParameters
        Model parameter set.
    time_grid_s : np.ndarray
        Output time grid in seconds.

    Returns
    -------
    FVSimulationResult
        Structured FV simulation output.
    """
    fv_class_edges, fv_pivot_points, fv_class_widths = create_fv_grid(params)

    fv_initial_density, fv_initial_inventory = create_fv_initial_condition(
        fv_pivot_points=fv_pivot_points,
        fv_class_widths=fv_class_widths,
        params=params,
    )

    fv_solution = solve_ivp(
        fun=fv_model_rhs,
        t_span=(time_grid_s[0], time_grid_s[-1]),
        y0=fv_initial_inventory,
        t_eval=time_grid_s,
        args=(params, fv_class_edges, fv_class_widths),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not fv_solution.success:
        raise RuntimeError(f"FV ODE solver failed: {fv_solution.message}")

    fv_inventory_history = fv_solution.y.T
    fv_density_history = fv_inventory_history / fv_class_widths[None, :]
    fv_total_number_history = np.sum(fv_inventory_history, axis=1)

    return FVSimulationResult(
        time_s=fv_solution.t,
        fv_class_edges=fv_class_edges,
        fv_pivot_points=fv_pivot_points,
        fv_class_widths=fv_class_widths,
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


def plot_moch_psd_evolution(moch_result: MOCHSimulationResult):
    """
    Plot the evolution of the PSD computed by MOCH.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.get_cmap("viridis", len(moch_result.time_s))

    for i, time_s in enumerate(moch_result.time_s):
        ax.plot(
            moch_result.moch_size_history[i, :],
            moch_result.moch_density_history[i, :],
            color=cmap(i),
            label=f"{time_s / 60.0:.1f} min",
        )

    ax.set_title("Evolution of the particle size distribution (MOCH)")
    ax.set_xlabel("Crystal size, $L$ [$\\mu$m]")
    ax.set_ylabel("Number density, $f(L,t)$")
    style_publication_axes(ax)
    ax.legend(title="Time", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return fig, ax


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

    ax.plot(
        time_s / 60.0,
        moch_total_number_history,
        marker="o",
        markersize=4,
        label="MOCH",
    )
    ax.plot(
        time_s / 60.0,
        fv_total_number_history,
        marker="s",
        markersize=4,
        label="Finite volume",
    )

    ax.axhline(
        total_number_initial,
        linestyle="--",
        linewidth=1.6,
        label="Initial total number",
    )

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


def plot_moch_characteristic_trajectories(moch_result: MOCHSimulationResult):
    """
    Plot the characteristic trajectories computed by MOCH.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for i in range(moch_result.moch_size_history.shape[1]):
        ax.plot(
            moch_result.time_s / 60.0,
            moch_result.moch_size_history[:, i],
            alpha=0.9,
        )

    ax.set_title("Characteristic trajectories (MOCH)")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Crystal size along characteristic, $L(t)$ [$\\mu$m]")
    style_publication_axes(ax)
    fig.tight_layout()
    return fig, ax


def plot_moch_numerical_vs_analytical(
    moch_result: MOCHSimulationResult,
    exact_size_history: np.ndarray,
    exact_density_history: np.ndarray,
):
    """
    Compare MOCH with the analytical solution at the final time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    final_index = -1
    final_time_min = moch_result.time_s[final_index] / 60.0

    axes[0].plot(
        moch_result.moch_size_history[final_index, :],
        label="Numerical MOCH",
    )
    axes[0].plot(
        exact_size_history[final_index, :],
        "--",
        label="Analytical",
    )
    axes[0].set_title(f"Characteristic positions at {final_time_min:.1f} min")
    axes[0].set_xlabel("Characteristic index")
    axes[0].set_ylabel("$L$ [$\\mu$m]")
    style_publication_axes(axes[0])
    axes[0].legend()

    axes[1].plot(
        moch_result.moch_density_history[final_index, :],
        label="Numerical MOCH",
    )
    axes[1].plot(
        exact_density_history[final_index, :],
        "--",
        label="Analytical",
    )
    axes[1].set_title(f"Number density at {final_time_min:.1f} min")
    axes[1].set_xlabel("Characteristic index")
    axes[1].set_ylabel("$f$")
    style_publication_axes(axes[1])
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def plot_moch_vs_fv_final_psd(
    moch_result: MOCHSimulationResult,
    fv_result: FVSimulationResult,
    exact_size_history: np.ndarray | None = None,
    exact_density_history: np.ndarray | None = None,
):
    """
    Compare the final-time PSD predicted by MOCH and FV.

    If available, the analytical solution is added as a reference.
    """
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    final_index = -1
    final_time_min = moch_result.time_s[final_index] / 60.0

    ax.plot(
        fv_result.fv_pivot_points,
        fv_result.fv_density_history[final_index, :],
        "-",
        linewidth=2.5,
        color="tab:blue",
        label="Finite volume",
    )

    ax.plot(
        moch_result.moch_size_history[final_index, :],
        moch_result.moch_density_history[final_index, :],
        "o",
        markersize=5,
        color="tab:orange",
        alpha=0.8,
        label="MOCH",
    )

    if exact_size_history is not None and exact_density_history is not None:
        ax.plot(
            exact_size_history[final_index, :],
            exact_density_history[final_index, :],
            "--",
            linewidth=1.5,
            color="black",
            label="Analytical",
        )

    ax.set_title(f"Comparison of numerical methods at $t = {final_time_min:.1f}$ min")
    ax.set_xlabel("Crystal size, $L$ [$\\mu$m]")
    ax.set_ylabel("Number density, $f(L,t)$")
    ax.set_xlim(
        np.min(fv_result.fv_pivot_points),
        np.max(moch_result.moch_size_history[final_index, :]) + 10.0,
    )
    style_publication_axes(ax)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Reporting
# =============================================================================

def print_simulation_summary(
    params: ModelParameters,
    moch_result: MOCHSimulationResult,
    fv_result: FVSimulationResult,
    exact_size_history: np.ndarray | None = None,
    exact_density_history: np.ndarray | None = None,
) -> None:
    """
    Print a compact simulation summary.
    """
    print(f"Initial total particle number: {params.total_number:.6f}")
    print(
        f"Final total particle number (MOCH): "
        f"{moch_result.moch_total_number_history[-1]:.6f}"
    )
    print(
        f"Final total particle number (FV):   "
        f"{fv_result.fv_total_number_history[-1]:.6f}"
    )

    if exact_size_history is not None and exact_density_history is not None:
        max_abs_error_size = np.max(
            np.abs(moch_result.moch_size_history[-1, :] - exact_size_history[-1, :])
        )
        max_abs_error_density = np.max(
            np.abs(
                moch_result.moch_density_history[-1, :]
                - exact_density_history[-1, :]
            )
        )

        print(
            f"Maximum absolute error in L at final time (MOCH): "
            f"{max_abs_error_size:.6e}"
        )
        print(
            f"Maximum absolute error in f at final time (MOCH): "
            f"{max_abs_error_density:.6e}"
        )


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    """
    Execute the MOCH-versus-FV comparison workflow.
    """
    apply_publication_plot_style()
    params = ModelParameters()

    # A common reporting grid is used so that both numerical solutions can be
    # compared at identical times without interpolation.
    time_grid_s = create_time_grid_seconds(params)

    # Solve the two numerical formulations.
    moch_result = solve_moch_simulation(params, time_grid_s)
    fv_result = solve_fv_simulation(params, time_grid_s)

    # Evaluate the analytical solution when the parameter regime admits a closed
    # form expression, here for p = 1.
    if np.isclose(params.p, 1.0):
        exact_size_history, exact_density_history = evaluate_analytical_solution_p1(
            time_grid_s=moch_result.time_s,
            moch_initial_grid=moch_result.moch_initial_grid,
            moch_initial_density=moch_result.moch_initial_density,
            params=params,
        )
    else:
        exact_size_history, exact_density_history = None, None

    print_simulation_summary(
        params=params,
        moch_result=moch_result,
        fv_result=fv_result,
        exact_size_history=exact_size_history,
        exact_density_history=exact_density_history,
    )

    # Visualizations
    plot_moch_psd_evolution(moch_result)
    plot_total_particle_number_comparison(
        time_s=moch_result.time_s,
        moch_total_number_history=moch_result.moch_total_number_history,
        fv_total_number_history=fv_result.fv_total_number_history,
        total_number_initial=params.total_number,
    )
    plot_moch_characteristic_trajectories(moch_result)

    if exact_size_history is not None and exact_density_history is not None:
        plot_moch_numerical_vs_analytical(
            moch_result=moch_result,
            exact_size_history=exact_size_history,
            exact_density_history=exact_density_history,
        )

    plot_moch_vs_fv_final_psd(
        moch_result=moch_result,
        fv_result=fv_result,
        exact_size_history=exact_size_history,
        exact_density_history=exact_density_history,
    )

    plt.show()


if __name__ == "__main__":
    main()