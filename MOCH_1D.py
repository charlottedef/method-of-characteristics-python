import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp
from scipy.stats import norm
from dataclasses import dataclass


# =============================================================================
# Population balance model solved by
# 1) the Method of Characteristics (MOCH)
# 2) a finite-volume (FV) discretization on a fixed size grid
# =============================================================================
#
# Governing equation
# ------------------
#     ∂f/∂t + ∂(G(L) f)/∂L = 0
#
# with growth law
#
#     G(L) = k_gS * (1 + gamma * L)^p
#
# The MOCH formulation tracks moving characteristic points L_i(t) together with
# their associated number density values f_i(t).
#
# The FV formulation follows the MATLAB structure supplied by the user:
#     1) define cell edges
#     2) define pivots (cell centers)
#     3) define cell widths
#     4) integrate the cell inventories Q_i(t)
#     5) recover the number density q_i(t) = Q_i(t) / ΔL_i
#
# For p = 1, the analytical solution is used as a reference for both numerical
# methods.
# =============================================================================


@dataclass(frozen=True)
class ModelParameters:
    """
    Container for all model and simulation parameters.

    Units
    -----
    growth_constant : µm/s
    mean_size       : µm
    std_dev         : µm
    class_width     : µm
    final_time_min  : min
    """
    growth_constant: float = 0.02   # Lumped growth constant k_g x S_g [µm/s]
    gamma: float = 0.005            # Size-dependence coefficient [- or 1/µm depending on formulation]
    p: float = 1.0                  # Growth exponent
    mean_size: float = 90.0         # Mean of the initial Gaussian PSD [µm]
    std_dev: float = 8.0            # Standard deviation of the initial Gaussian PSD [µm]
    class_width: float = 1.5        # Initial spacing between characteristic points [µm]
    total_number: float = 5000.0    # Total initial number of particles
    final_time_min: float = 100.0   # Final simulation time [min]
    n_time_points: int = 10         # Number of reported time points


def create_initial_grid(params: ModelParameters) -> np.ndarray:
    """
    Construct the initial size grid.
    The grid spans ±4 standard deviations around the mean size.
    Formally, the grid is

    L0 = {μ - 4σ, μ - 4σ + ΔL, μ - 4σ + 2ΔL, ..., μ + 4σ}

    where ΔL = class_width and N is chosen such that the grid covers the
    range up to approximately μ + 4σ.
    """
    span = 4.0 * params.std_dev
    start = params.mean_size - span
    stop = params.mean_size + span

    # The small addition ensures that the nominal endpoint is included.
    L0 = np.arange(start, stop + 0.5 * params.class_width, params.class_width)
    return L0

def create_extended_fv_grid(params: ModelParameters, max_size: float = 450.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt ein erweitertes, ortsfestes Gitter für die Eulersche Finite-Volumen-Methode.
    """
    span = 4.0 * params.std_dev
    start = params.mean_size - span
    
    # Gitter bis zur erwarteten Maximalgröße aufspannen
    L_grid_fv = np.arange(start, max_size + 0.5 * params.class_width, params.class_width)
    
    # Initiale Verteilung auf das neue, große Gitter projizieren
    f_initial_fv = create_initial_number_density(L_grid_fv, params)
    
    return L_grid_fv, f_initial_fv

def create_initial_number_density(L0: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Construct the initial Gaussian number density and normalize it such that the
    discrete integral equals the prescribed total number of particles.

    The normalization follows the same idea as in the MATLAB code:
        1) compute the Gaussian density values
        2) normalize the discrete integral to one
        3) scale to the desired total particle number
    """
    raw_density = norm.pdf(L0, loc=params.mean_size, scale=params.std_dev)

    # Convert the sampled Gaussian into a number density whose discrete integral
    # over size equals the requested total particle number.
    f0 = raw_density / (np.sum(raw_density) * params.class_width) * params.total_number
    return f0


def build_initial_state(L0: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """
    Assemble the ODE initial state vector.

    State ordering:
        y = [L_1, ..., L_N, f_1, ..., f_N]
    """
    return np.concatenate([L0, f0])


def growth_model_rhs(
    t: float,
    y: np.ndarray,
    growth_constant: float,
    gamma: float,
    p: float,
    n_classes: int
) -> np.ndarray:
    """
    Right-hand side of the ODE system for the method of characteristics.

    Parameters
    ----------
    t : float
        Time [s].
    y : ndarray
        Current state vector containing [L, f].
    growth_constant : float
        Lumped growth constant k_g S_g [µm/s].
    gamma : float
        Size-dependence coefficient in the growth law.
    p : float
        Growth exponent.
    n_classes : int
        Number of characteristic points (initial size classes).

    Returns
    -------
    dydt : ndarray
        Time derivative of the state vector.
    """
    # The equations below come from the population balance equation
    # ∂f/∂t + ∂(Gf)/∂L = h, rewritten with the product rule and transformed with the method of characteristics.
    # Each characteristic tracks one size point L, which moves according to dL/dt = G(L,t).
    # The associated number density changes along that path as df/dt = -f·∂G/∂L + h.
    # Here, h = 0 because the model only includes growth and neglects additional source terms such as nucleation, aggregation, and breakage.
    # For the growth law G(L) = k(1 + γL)^p, differentiating with respect to L gives ∂G/∂L = kγp(1 + γL)^(p-1), which yields the implemented RHS expressions.
    # Split the state vector into characteristic positions and densities.
    L = y[:n_classes]
    f = y[n_classes:]

    # Characteristic motion in size space.
    dL_dt = growth_constant * (1.0 + gamma * L) ** p

    # Number density evolution along the characteristic.
    df_dt = -f * growth_constant * gamma * p * (1.0 + gamma * L) ** (p - 1.0)

    return np.concatenate([dL_dt, df_dt])

def fv_model_rhs(
    t: float,
    N_temp: np.ndarray,
    growth_constant: float,
    gamma: float,
    p: float,
    L_centers: np.ndarray,
    delta_L: float
) -> np.ndarray:
    """
    Right-hand side of the ODE system for the Finite Volume method.
    Structure mirrors the provided MATLAB implementation.
    """
    n_classes = len(L_centers)
    N_temp = np.maximum(N_temp, 0.0)
    
    # Pad array for boundary conditions
    n_temp = np.zeros(n_classes + 2)
    n_temp[1:-1] = N_temp / delta_L

    # Growth rate at cell boundaries
    L_faces = L_centers + 0.5 * delta_L
    G_faces = growth_constant * (1.0 + gamma * L_faces)**p
    G1_pos = np.zeros(n_classes + 1)
    G1_pos[1:] = G_faces
    G1_pos[-1] = 0.0 

    epsilon = 1e-10

    # Flux limiter calculation (Koren/Van Leer analogy)
    num = n_temp[2:] - n_temp[1:-1] + epsilon
    den = n_temp[1:-1] - n_temp[:-2] + epsilon
    r = num / den
    PHI = (np.abs(r) + r) / (1.0 + np.abs(r))

    n_at_borders_pos = np.zeros(n_classes + 1)
    n_at_borders_pos[1:] = n_temp[1:-1] + 0.5 * PHI * den
    
    Y1_at_borders_pos = np.maximum(n_at_borders_pos, 0.0)

    # Flux computation
    dNdt_Growth1 = G1_pos * Y1_at_borders_pos
    dNdt_Growth1_out = -dNdt_Growth1[1:]
    dNdt_Growth1_in = dNdt_Growth1[:-1]

    dNdt = dNdt_Growth1_out + dNdt_Growth1_in
    return dNdt

def compute_effective_class_widths(L: np.ndarray) -> np.ndarray:
    """
    Compute the effective class widths from the current characteristic locations.

    This step is important because the characteristic points move with time.
    Therefore, the spacing in size space is no longer constant, and the local
    integration width for each class must be updated before computing the total
    particle number from the number density.

    The interior widths are defined by half the distance to the left neighbor
    plus half the distance to the right neighbor. The boundary widths are taken
    as the nearest-neighbor spacing.
    """
    widths = np.zeros_like(L)

    widths[1:-1] = 0.5 * (L[1:-1] - L[:-2]) + 0.5 * (L[2:] - L[1:-1])
    widths[0] = L[1] - L[0]
    widths[-1] = L[-1] - L[-2]

    return widths


def analytical_solution_p1(
    t: np.ndarray,
    L0: np.ndarray,
    f0: np.ndarray,
    growth_constant: float,
    gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical solution for the special case p = 1.

    From the literature:
        L(t) = ((1 + gamma*L0) * exp(gamma*k*t) - 1) / gamma
        f(t) = f0(L0) * exp(-gamma*k*t)

    Parameters
    ----------
    t : ndarray
        Time points [s].
    L0 : ndarray
        Initial characteristic locations [µm].
    f0 : ndarray
        Initial number density values.
    growth_constant : float
        Lumped growth constant [µm/s].
    gamma : float
        Size-dependence coefficient.

    Returns
    -------
    L_exact : ndarray
        Exact characteristic positions with shape (n_times, n_classes).
    f_exact : ndarray
        Exact densities with shape (n_times, n_classes).
    """
    t = np.asarray(t)
    exp_forward = np.exp(gamma * growth_constant * t)[:, None]
    exp_decay = np.exp(-gamma * growth_constant * t)[:, None]

    L_exact = ((1.0 + gamma * L0[None, :]) * exp_forward - 1.0) / gamma
    f_exact = f0[None, :] * exp_decay

    return L_exact, f_exact


def simulate_moch(params: ModelParameters):
    """
    Run the MOCH simulation and return all relevant outputs.
    """
    # -------------------------------------------------------------------------
    # 1) Build initial characteristic grid and initial number density
    # -------------------------------------------------------------------------
    L_initial = create_initial_grid(params)
    f_initial = create_initial_number_density(L_initial, params)
    y0 = build_initial_state(L_initial, f_initial)

    n_classes = L_initial.size

    # -------------------------------------------------------------------------
    # 2) Define the time grid
    # -------------------------------------------------------------------------
    # The growth constant is given in µm/s, so the ODE system is solved in seconds.
    t_eval = np.linspace(0.0, params.final_time_min * 60.0, params.n_time_points)
    t_span = (t_eval[0], t_eval[-1])

    # -------------------------------------------------------------------------
    # 3) Integrate the characteristic ODE system
    # -------------------------------------------------------------------------
    solution = solve_ivp(
        fun=growth_model_rhs,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(params.growth_constant, params.gamma, params.p, n_classes),
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    # solve_ivp returns solution.y with shape (n_states, n_times).
    # For easier post-processing, transpose it to (n_times, n_states).
    Y = solution.y.T
    T = solution.t

    # Extract the characteristic positions and number density history.
    L_history = Y[:, :n_classes]
    f_history = Y[:, n_classes:]

    # -------------------------------------------------------------------------
    # 4) Compute the total particle number over time
    # -------------------------------------------------------------------------
    total_number_history = np.zeros_like(T)

    for i in range(len(T)):
        widths = compute_effective_class_widths(L_history[i, :])
        total_number_history[i] = np.sum(f_history[i, :] * widths)

    return T, L_initial, f_initial, L_history, f_history, total_number_history

def simulate_fv(params: ModelParameters, L_initial: np.ndarray, f_initial: np.ndarray, T: np.ndarray):
    """
    Run the Finite Volume simulation using the identical time grid.
    """
    # Transform density f to absolute numbers N for the FV solver
    N_initial = f_initial * params.class_width
    t_span = (T[0], T[-1])

    solution = solve_ivp(
        fun=fv_model_rhs,
        t_span=t_span,
        y0=N_initial,
        t_eval=T,
        args=(params.growth_constant, params.gamma, params.p, L_initial, params.class_width),
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(f"FV ODE solver failed: {solution.message}")

    N_history = solution.y.T
    
    # Transform back to number density
    f_history_fv = N_history / params.class_width
    
    # Neu: Berechnung der Gesamtpartikelzahl über alle Klassen für jeden Zeitschritt
    total_number_fv = np.sum(f_history_fv * params.class_width, axis=1)
    
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


def plot_psd_evolution(T, L_history, f_history):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.get_cmap("viridis", len(T))

    for i, t in enumerate(T):
        label = f"{t / 60.0:.1f} min"
        ax.plot(L_history[i, :], f_history[i, :], color=cmap(i), label=label)

    ax.set_title("Evolution of the particle size distribution (MOCH)")
    ax.set_xlabel("Crystal size, $L$ [$\\mu$m]")
    ax.set_ylabel("Number density, $f(L,t)$")
    style_axes(ax)
    ax.legend(title="Time", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_total_number(T, total_number_moch, total_number_fv, total_number_initial):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    ax.plot(T / 60.0, total_number_moch, marker="o", markersize=4, label="Computed total number (MOCH)")
    ax.plot(T / 60.0, total_number_fv, marker="s", markersize=4, label="Computed total number (Finite Volume)")
    
    ax.axhline(total_number_initial, linestyle="--", linewidth=1.6, label="Initial total number")

    min_particles = min(np.min(total_number_moch), np.min(total_number_fv))
    max_particles = max(np.max(total_number_moch), np.max(total_number_fv))

    y_lower = np.floor(min_particles) - 1
    y_upper = np.ceil(max_particles) + 1
    ax.set_ylim(y_lower, y_upper)
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_title("Total particle number over time")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Total particle number")

    style_axes(ax)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_characteristics(T, L_history):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for i in range(L_history.shape[1]):
        ax.plot(T / 60.0, L_history[:, i], alpha=0.9)

    ax.set_title("Characteristic trajectories (MOCH)")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Crystal size along characteristic, $L(t)$ [$\\mu$m]")
    style_axes(ax)
    fig.tight_layout()
    return fig, ax


def plot_numerical_vs_analytical(T, L_history, f_history, L_exact, f_exact):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    i_final = -1
    final_time_min = T[i_final] / 60.0

    axes[0].plot(L_history[i_final, :], label="Numerical MOCH")
    axes[0].plot(L_exact[i_final, :], "--", label="Analytical")
    axes[0].set_title(f"Characteristic positions at {final_time_min:.1f} min")
    axes[0].set_xlabel("Characteristic index")
    axes[0].set_ylabel("$L$ [$\\mu$m]")
    style_axes(axes[0])
    axes[0].legend()

    axes[1].plot(f_history[i_final, :], label="Numerical MOCH")
    axes[1].plot(f_exact[i_final, :], "--", label="Analytical")
    axes[1].set_title(f"Number density at {final_time_min:.1f} min")
    axes[1].set_xlabel("Characteristic index")
    axes[1].set_ylabel("$f$")
    style_axes(axes[1])
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def plot_method_comparison(T, L_history_moch, f_history_moch, L_grid_fv, f_history_fv, L_exact, f_exact):
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    i_final = -1
    final_time_min = T[i_final] / 60.0

    ax.plot(L_grid_fv, f_history_fv[i_final, :], '-', color='tab:blue', label='Finite Volume', linewidth=2.5)
    ax.plot(L_history_moch[i_final, :], f_history_moch[i_final, :], 'o', color='tab:orange', markersize=5, label='MOCH', alpha=0.8)

    if L_exact is not None and f_exact is not None:
        ax.plot(L_exact[i_final, :], f_exact[i_final, :], '--', color='black', label='Analytical', linewidth=1.5)

    ax.set_title(f"Comparison of Numerical Methods at $t = {final_time_min:.1f}$ min")
    ax.set_xlabel("Crystal size, $L$ [$\\mu$m]")
    ax.set_ylabel("Number density, $f(L,t)$")
    
    ax.set_xlim(np.min(L_grid_fv), np.max(L_history_moch[i_final, :]) + 10)
    
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def main():
    apply_publication_style()
    params = ModelParameters()

    # 1. Simulate MOCH (Lagrange)
    T, L_initial, f_initial, L_history_moch, f_history_moch, total_number_moch = simulate_moch(params)
    total_number_initial = np.sum(f_initial * params.class_width)

    # 2. Simulate Finite Volume (Euler) auf erweitertem Gitter
    L_grid_fv, f_initial_fv = create_extended_fv_grid(params, max_size=450.0)
    f_history_fv, total_number_fv = simulate_fv(params, L_grid_fv, f_initial_fv, T)

    # 3. Compute Analytical Solution (for p=1)
    if np.isclose(params.p, 1.0):
        L_exact, f_exact = analytical_solution_p1(
            t=T,
            L0=L_initial,
            f0=f_initial,
            growth_constant=params.growth_constant,
            gamma=params.gamma
        )
        
        max_abs_error_L = np.max(np.abs(L_history_moch[-1, :] - L_exact[-1, :]))
        max_abs_error_f = np.max(np.abs(f_history_moch[-1, :] - f_exact[-1, :]))

        print(f"Initial total particle number: {total_number_initial:.6f}")
        print(f"Final total particle number (MOCH): {total_number_moch[-1]:.6f}")
        print(f"Final total particle number (FV):   {total_number_fv[-1]:.6f}")
        print(f"Maximum absolute error in L at final time (MOCH): {max_abs_error_L:.6e}")
        print(f"Maximum absolute error in f at final time (MOCH): {max_abs_error_f:.6e}")
    else:
        L_exact, f_exact = None, None
        print(f"Initial total particle number: {total_number_initial:.6f}")
        print(f"Final total particle number (MOCH): {total_number_moch[-1]:.6f}")
        print(f"Final total particle number (FV):   {total_number_fv[-1]:.6f}")

    # 4. Visualizations
    plot_psd_evolution(T, L_history_moch, f_history_moch)
    plot_total_number(T, total_number_moch, total_number_fv, total_number_initial)
    plot_characteristics(T, L_history_moch)
    
    if L_exact is not None and f_exact is not None:
        plot_numerical_vs_analytical(T, L_history_moch, f_history_moch, L_exact, f_exact)
        
    plot_method_comparison(T, L_history_moch, f_history_moch, L_grid_fv, f_history_fv, L_exact, f_exact)
    
    plt.show()


if __name__ == "__main__":
    main()