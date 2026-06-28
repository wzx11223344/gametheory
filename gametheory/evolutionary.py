"""
Evolutionary Game Theory Module
===============================

Provides implementations of evolutionary game theory concepts, including
continuous and discrete replicator dynamics, evolutionarily stable
strategy (ESS) detection, and classic evolutionary games.

Algorithms implemented:
- Continuous replicator dynamics
- Discrete replicator dynamics
- Evolutionarily Stable Strategy (ESS) detection
- Hawk-Dove game
- Prisoner's Dilemma as evolutionary game
- Phase diagram plotting
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations


def replicator_dynamics(A, x0, T=100, dt=0.01):
    """Continuous replicator dynamics.

    dx_i/dt = x_i * ((Ax)_i - x^T A x)

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Payoff matrix (symmetric game).
    x0 : ndarray, shape (n,)
        Initial population state (must sum to 1).
    T : float
        Total time.
    dt : float
        Time step.

    Returns
    -------
    tuple
        (t_values, x_trajectory) where x_trajectory has shape (n_steps, n).
    """
    A = np.asarray(A, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    x0 = x0 / x0.sum()

    n = len(x0)
    n_steps = int(T / dt) + 1
    t_values = np.linspace(0, T, n_steps)
    trajectory = np.zeros((n_steps, n))
    trajectory[0] = x0

    x = x0.copy()
    for step in range(1, n_steps):
        Ax = A @ x
        avg_payoff = x @ Ax

        # dx_i/dt = x_i * ((Ax)_i - avg_payoff)
        dx = x * (Ax - avg_payoff)
        x = x + dt * dx

        # Ensure non-negativity and normalization
        x = np.maximum(x, 0)
        s = x.sum()
        if s > 1e-12:
            x = x / s
        else:
            x[:] = 1.0 / n

        trajectory[step] = x

    return t_values, trajectory


def discrete_replicator(A, x0, n_generations=100):
    """Discrete-time replicator dynamics.

    x_i(t+1) = x_i(t) * (Ax)_i / (x^T A x)

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Payoff matrix (symmetric game).
    x0 : ndarray, shape (n,)
        Initial population state (must sum to 1).
    n_generations : int
        Number of generations.

    Returns
    -------
    ndarray
        Population trajectory, shape (n_generations + 1, n).
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x0, dtype=float)
    x = x / x.sum()

    n = len(x)
    trajectory = np.zeros((n_generations + 1, n))
    trajectory[0] = x

    for g in range(n_generations):
        Ax = A @ x
        avg_payoff = x @ Ax

        if abs(avg_payoff) < 1e-12:
            x_new = x * (1.0 + Ax)
        else:
            x_new = x * Ax / avg_payoff

        x_new = np.maximum(x_new, 0)
        s = x_new.sum()
        if s > 1e-12:
            x = x_new / s

        trajectory[g + 1] = x

    return trajectory


def find_ESS(A, tol=1e-6):
    """Find Evolutionarily Stable Strategies for a symmetric game.

    A strategy x is an ESS if:
    1. It is a Nash equilibrium: x^T A x >= y^T A x for all y
    2. Stability: if x^T A x = y^T A x then x^T A y > y^T A y

    This implementation checks pure strategies and the interior mixed
    equilibrium for 2x2 games, and uses support enumeration for larger games.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric payoff matrix.
    tol : float
        Numerical tolerance.

    Returns
    -------
    list of ndarray
        List of ESS strategy vectors.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    ess_list = []

    # Check pure strategies
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1.0

        if _is_ess(A, x, tol):
            ess_list.append(x.copy())

    # For 2x2 games, check mixed equilibrium
    if n == 2:
        denom = A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1]
        if abs(denom) > tol:
            p = (A[1, 1] - A[0, 1]) / denom
            if 0 < p < 1:
                x = np.array([p, 1 - p])
                if _is_ess(A, x, tol):
                    ess_list.append(x)

    # For larger games, enumerate support profiles
    if n > 2:
        for k in range(1, min(n, 5)):
            for support in combinations(range(n), k):
                x = _find_mixed_on_support(A, list(support))
                if x is not None and _is_ess(A, x, tol):
                    # Check for duplicates
                    is_dup = False
                    for existing in ess_list:
                        if np.allclose(existing, x, atol=tol):
                            is_dup = True
                            break
                    if not is_dup:
                        ess_list.append(x)

    return ess_list


def _is_ess(A, x, tol=1e-6):
    """Check if strategy x is an Evolutionarily Stable Strategy.

    Parameters
    ----------
    A : ndarray
        Payoff matrix.
    x : ndarray
        Candidate strategy.
    tol : float
        Numerical tolerance.

    Returns
    -------
    bool
        True if x is an ESS.
    """
    n = len(x)
    x_payoff = x @ A @ x

    # Condition 1: x must be a Nash equilibrium against itself
    for i in range(n):
        y = np.zeros(n)
        y[i] = 1.0
        if y @ A @ x > x_payoff + tol:
            return False

    # Condition 2: For any alternative best response y != x
    best_responses = []
    for i in range(n):
        y = np.zeros(n)
        y[i] = 1.0
        if abs(y @ A @ x - x_payoff) <= tol:
            best_responses.append(y)

    for y_pure in best_responses:
        if not np.allclose(y_pure, x, atol=tol):
            # Check: x^T A y > y^T A y
            if x @ A @ y_pure <= y_pure @ A @ y_pure + tol:
                return False

    return True


def _find_mixed_on_support(A, support):
    """Find a mixed strategy equilibrium on given support for symmetric game.

    Parameters
    ----------
    A : ndarray
        Payoff matrix.
    support : list of int
        Strategy indices in support.

    Returns
    -------
    ndarray or None
        Mixed strategy on the support, or None if no equilibrium.
    """
    n = A.shape[0]
    k = len(support)

    if k == 1:
        x = np.zeros(n)
        x[support[0]] = 1.0
        return x

    # Solve: all strategies in support give equal payoff
    # (A x)_i = v for all i in support, sum(x) = 1
    M = np.zeros((k, k))
    rhs = np.zeros(k)

    # Indifference equations
    ref = support[0]
    for idx in range(1, k):
        i = support[idx]
        for jdx, j in enumerate(support):
            M[idx - 1, jdx] = A[i, j] - A[ref, j]
        rhs[idx - 1] = 0

    # Sum to 1
    M[k - 1, :] = 1.0
    rhs[k - 1] = 1.0

    try:
        sol = np.linalg.solve(M, rhs)
        if np.any(sol < -1e-8):
            return None
        sol = np.maximum(sol, 0)
        sol = sol / sol.sum()

        x = np.zeros(n)
        for idx, j in enumerate(support):
            x[j] = sol[idx]
        return x
    except np.linalg.LinAlgError:
        return None


def hawk_dove_game(v=2.0, c=3.0):
    """Classic Hawk-Dove (Snowdrift) evolutionary game.

    Strategies: Hawk (H) = 0, Dove (D) = 1.
    Payoffs:
    - H vs H: ((v-c)/2, (v-c)/2)
    - H vs D: (v, 0)
    - D vs H: (0, v)
    - D vs D: (v/2, v/2)

    The mixed ESS is: play Hawk with probability v/c.

    Parameters
    ----------
    v : float
        Value of the resource.
    c : float
        Cost of fighting.

    Returns
    -------
    tuple
        (payoff_matrix, mixed_ess, pure_ess_list)
    """
    A = np.array([
        [(v - c) / 2, v],
        [0, v / 2]
    ])

    mixed_ess = np.array([v / c, 1 - v / c]) if c > 0 else np.array([1.0, 0.0])
    pure_ess = find_ESS(A)

    return A, mixed_ess, pure_ess


def prisoners_dilemma(b=3.0, c=1.0):
    """Prisoner's Dilemma as an evolutionary game.

    Strategies: Cooperate (C) = 0, Defect (D) = 1.
    Payoffs (benefit b, cost c, b > c > 0):
    - C vs C: (b-c, b-c)
    - C vs D: (-c, b)
    - D vs C: (b, -c)
    - D vs D: (0, 0)

    In evolutionary setting, Defect is the only ESS.

    Parameters
    ----------
    b : float
        Benefit of cooperation (must be > c).
    c : float
        Cost of cooperation.

    Returns
    -------
    tuple
        (payoff_matrix, ess_list)
    """
    A = np.array([
        [b - c, -c],
        [b, 0]
    ])

    ess_list = find_ESS(A)
    return A, ess_list


def plot_dynamics(trajectories, labels=None, title="Replicator Dynamics",
                  save_path=None):
    """Plot phase diagram for replicator dynamics trajectories.

    For 2-strategy games, plots the proportion of strategy 0 over time.
    For 3-strategy games, plots a ternary (simplex) phase diagram.

    Parameters
    ----------
    trajectories : list of tuple
        Each element is (t, x) from replicator_dynamics, or an ndarray
        from discrete_replicator.
    labels : list of str, optional
        Labels for each trajectory.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    first_traj = trajectories[0]
    if isinstance(first_traj, tuple):
        n_strategies = first_traj[1].shape[1]
    else:
        n_strategies = first_traj.shape[1]

    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]

    # Time series plot (left)
    ax1 = axes[0]
    for idx, traj in enumerate(trajectories):
        if isinstance(traj, tuple):
            t, x = traj
        else:
            x = traj
            t = np.arange(len(x))
        for s in range(n_strategies):
            ax1.plot(t, x[:, s], alpha=0.7,
                     label=f"{labels[idx]} - S{s}" if s == 0 else f"__nolabel__")

    ax1.set_xlabel("Time / Generations")
    ax1.set_ylabel("Population Share")
    ax1.set_title("Population Dynamics Over Time")
    ax1.legend(loc='best', fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Phase plot (right): S1 vs S0 for 2-strategy
    ax2 = axes[1]
    if n_strategies == 2:
        for idx, traj in enumerate(trajectories):
            if isinstance(traj, tuple):
                _, x = traj
            else:
                x = traj
            ax2.plot(x[:, 0], x[:, 1], '-o', markersize=2, alpha=0.7,
                     label=labels[idx])
        ax2.set_xlabel("Strategy 0 Share")
        ax2.set_ylabel("Strategy 1 Share")
        ax2.set_title("Phase Portrait")
        ax2.plot([0, 1], [1, 0], 'k--', alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
    elif n_strategies == 3:
        # Simplex plot for 3 strategies
        for idx, traj in enumerate(trajectories):
            if isinstance(traj, tuple):
                _, x = traj
            else:
                x = traj
            ax2.plot(x[:, 0], x[:, 1], '-o', markersize=2, alpha=0.7,
                     label=labels[idx])

        # Draw simplex triangle
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3) / 2, 0]
        ax2.plot(triangle_x, triangle_y, 'k-', alpha=0.5)
        ax2.set_xlabel("Strategy 0")
        ax2.set_ylabel("Strategy 1")
        ax2.set_title("Simplex Phase Portrait")
        ax2.set_aspect('equal')
    else:
        # For >3 strategies: plot all components
        for idx, traj in enumerate(trajectories):
            if isinstance(traj, tuple):
                _, x = traj
            else:
                x = traj
            ax2.plot(x[:, 0], x[:, 1], '-o', markersize=2, alpha=0.7,
                     label=labels[idx])
        ax2.set_xlabel("Strategy 0 Share")
        ax2.set_ylabel("Strategy 1 Share")
        ax2.set_title("Phase Portrait (S0 vs S1)")

    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
