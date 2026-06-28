"""
Bargaining Theory Module
========================

Provides implementations of classic bargaining solutions, including
Rubinstein alternating-offers, Nash bargaining solution (axiomatic),
Kalai-Smorodinsky, and egalitarian solutions.

Algorithms implemented:
- Rubinstein alternating-offers bargaining
- Nash bargaining solution (axiomatic)
- Kalai-Smorodinsky bargaining solution
- Egalitarian bargaining solution
"""

import numpy as np
from scipy.optimize import minimize


def rubinstein_bargaining(v, delta1, delta2, T=None):
    """Rubinstein alternating-offers bargaining game.

    Two players bargain over a surplus v. Player 1 makes the first offer.
    Offers alternate. If no agreement by period T, both get 0.

    Parameters
    ----------
    v : float
        Total surplus to divide.
    delta1 : float
        Discount factor for player 1 (in [0, 1]).
    delta2 : float
        Discount factor for player 2 (in [0, 1]).
    T : int, optional
        Maximum number of periods. If None, infinite horizon.

    Returns
    -------
    tuple
        (share1, share2) - equilibrium shares for each player.
    """
    if T is None or T >= 100000:
        # Infinite horizon: unique SPE
        # Player 1's share: (1 - delta2) / (1 - delta1 * delta2)
        share1 = v * (1 - delta2) / (1 - delta1 * delta2)
        share2 = v - share1
    else:
        # Finite horizon: solve by backward induction
        if T == 1:
            # Player 1 makes final offer: takes everything
            share1 = v
            share2 = 0
        else:
            # Backward induction
            s = v  # value for player making last offer at stage T
            for t in range(T - 1, 0, -1):
                if t % 2 == 1:
                    # Player 1's turn
                    s = max(0, v - delta2 * s)
                else:
                    # Player 2's turn
                    s = max(0, v - delta1 * s)

            if T % 2 == 1:
                share1 = s
                share2 = v - s
            else:
                share2 = s
                share1 = v - s

    return share1, share2


def nash_bargaining_solution(utilities, disagreement=(0, 0)):
    """Nash bargaining solution (axiomatic).

    Maximizes the product of surplus utilities over the disagreement point.
    (u1 - d1) * (u2 - d2) subject to u1 + u2 <= total_surplus.

    Parameters
    ----------
    utilities : array-like
        Feasible utility pairs as list of (u1, u2) tuples,
        or a callable constraint describing the feasible set.
        If array-like, finds the point that maximizes Nash product.
    disagreement : tuple
        Disagreement point (d1, d2).

    Returns
    -------
    ndarray
        Nash bargaining solution (u1, u2).
    """
    d1, d2 = disagreement

    if callable(utilities):
        # utilities is a constraint: find max Nash product on feasible set
        def neg_nash_product(x):
            u1, u2 = x
            if utilities(u1, u2) > 0:  # constraint violated
                return 1e10
            # Ensure above disagreement with buffer
            if u1 < d1 - 1e-6 or u2 < d2 - 1e-6:
                return 1e10
            nash = (u1 - d1) * (u2 - d2)
            return -nash if nash > 0 else 1e10

        result = minimize(
            neg_nash_product,
            x0=[d1 + 0.5, d2 + 0.5],
            method='Nelder-Mead',
            options={'maxiter': 10000}
        )
        return result.x

    # Discrete set of feasible pairs
    utilities = np.asarray(utilities)
    best_val = -np.inf
    best_pair = None

    for pair in utilities:
        u1, u2 = pair
        if u1 >= d1 and u2 >= d2:
            nash_val = (u1 - d1) * (u2 - d2)
            if nash_val > best_val:
                best_val = nash_val
                best_pair = pair

    if best_pair is None:
        best_pair = np.array([d1, d2])

    return np.asarray(best_pair)


def kalai_smorodinsky(utilities, ideal_point):
    """Kalai-Smorodinsky bargaining solution.

    Finds the maximal point on the Pareto frontier that maintains the
    ratio of gains equal to the ratio of ideal gains.

    For each feasible point (u1, u2) with ideal (I1, I2) and
    disagreement (d1, d2):
        (u1 - d1) / (I1 - d1) = (u2 - d2) / (I2 - d2)

    Parameters
    ----------
    utilities : ndarray
        Array of feasible utility pairs, shape (n, 2).
    ideal_point : tuple
        Ideal point (I1, I2) - maximum utility each player could get.

    Returns
    -------
    ndarray
        Kalai-Smorodinsky solution (u1, u2).
    """
    utilities = np.asarray(utilities)
    I1, I2 = ideal_point

    if I1 <= 0 or I2 <= 0:
        return utilities[np.argmax(utilities[:, 0] + utilities[:, 1])]

    best_val = -np.inf
    best_pair = None

    for pair in utilities:
        u1, u2 = pair
        if u1 >= 0 and u2 >= 0 and u1 <= I1 and u2 <= I2:
            ratio1 = u1 / I1 if I1 > 0 else 0
            ratio2 = u2 / I2 if I2 > 0 else 0
            min_ratio = min(ratio1, ratio2)
            if min_ratio > best_val:
                # Check if close to the KS line
                if abs(ratio1 - ratio2) < 0.05 or min_ratio > 0.95:
                    best_val = min_ratio
                    best_pair = pair

    if best_pair is None:
        # Fallback: find Pareto optimal point closest to proportional
        best_pair = utilities[np.argmax(
            np.minimum(utilities[:, 0] / max(I1, 1e-10),
                       utilities[:, 1] / max(I2, 1e-10))
        )]

    return np.asarray(best_pair)


def egalitarian_solution(utilities):
    """Egalitarian bargaining solution.

    Maximizes the minimum utility across players (Rawlsian).
    max min(u1, u2) over all feasible points.

    Parameters
    ----------
    utilities : ndarray
        Array of feasible utility pairs, shape (n, 2).

    Returns
    -------
    ndarray
        Egalitarian solution (u1, u2).
    """
    utilities = np.asarray(utilities)
    min_utils = np.min(utilities, axis=1)
    best_idx = np.argmax(min_utils)
    return utilities[best_idx]
