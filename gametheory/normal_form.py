"""
Normal-Form Game Module
=======================

Provides classes and algorithms for normal-form (strategic-form) games,
including Nash equilibrium computation via multiple methods, correlated
equilibrium, best response, and equilibrium verification.

Algorithms implemented:
- Brute-force support enumeration for Nash equilibrium
- Lemke-Howson algorithm for bimatrix games
- Correlated equilibrium via linear programming
- Closed-form 2x2 mixed Nash
- Fictitious play for large games
"""

import numpy as np
from itertools import combinations, product
from scipy.optimize import linprog


class NormalFormGame:
    """Two-player normal-form game with payoff matrices.

    Parameters
    ----------
    A : ndarray, shape (n_actions_1, n_actions_2)
        Payoff matrix for player 1 (row player).
    B : ndarray, shape (n_actions_1, n_actions_2), optional
        Payoff matrix for player 2 (column player). If None, assumes zero-sum.
    player_names : list of str, optional
        Names for players.

    Attributes
    ----------
    A : ndarray
        Player 1's payoff matrix.
    B : ndarray
        Player 2's payoff matrix.
    n_actions_1 : int
        Number of actions for player 1.
    n_actions_2 : int
        Number of actions for player 2.
    """

    def __init__(self, A, B=None, player_names=None):
        self.A = np.asarray(A, dtype=float)
        if B is None:
            self.B = -self.A.copy()
        else:
            self.B = np.asarray(B, dtype=float)
        self.n_actions_1 = self.A.shape[0]
        self.n_actions_2 = self.A.shape[1]
        if player_names is None:
            player_names = ["Player 1", "Player 2"]
        self.player_names = player_names

    def best_response(self, strategies, player=1):
        """Compute the best response of a player to a strategy profile.

        Parameters
        ----------
        strategies : tuple of ndarray
            Current strategy profile (sigma1, sigma2).
        player : int
            Player for whom to compute best response (1 or 2).

        Returns
        -------
        ndarray
            Pure strategy best response indices.
        """
        sigma1, sigma2 = strategies
        if player == 1:
            payoffs = self.A @ sigma2
            max_val = np.max(payoffs)
            return np.where(np.abs(payoffs - max_val) < 1e-10)[0]
        else:
            payoffs = sigma1 @ self.B
            max_val = np.max(payoffs)
            return np.where(np.abs(payoffs - max_val) < 1e-10)[0]

    def is_nash(self, strategies, tol=1e-6):
        """Verify whether a strategy profile is a Nash equilibrium.

        Parameters
        ----------
        strategies : tuple of ndarray
            Strategy profile (sigma1, sigma2).
        tol : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if the strategy profile is a Nash equilibrium.
        """
        sigma1, sigma2 = strategies
        # Check that strategies are valid probability distributions
        if not (np.abs(sigma1.sum() - 1.0) < tol and np.all(sigma1 >= -tol)):
            return False
        if not (np.abs(sigma2.sum() - 1.0) < tol and np.all(sigma2 >= -tol)):
            return False

        # Compute payoffs for each pure strategy
        p1_payoffs = self.A @ sigma2
        p1_best_val = np.max(p1_payoffs)
        # Player 1 only puts positive probability on best responses
        for i in range(self.n_actions_1):
            if sigma1[i] > tol and p1_payoffs[i] < p1_best_val - tol:
                return False

        p2_payoffs = sigma1 @ self.B
        p2_best_val = np.max(p2_payoffs)
        for j in range(self.n_actions_2):
            if sigma2[j] > tol and p2_payoffs[j] < p2_best_val - tol:
                return False

        return True

    def find_nash_brute_force(self):
        """Find all Nash equilibria by enumerating support profiles.

        For each possible support pair, solves a linear system. Only
        practical for small games (n_actions <= 6 or so).

        Returns
        -------
        list of tuple
            List of Nash equilibria, each as (sigma1, sigma2).
        """
        equilibria = []
        n1, n2 = self.n_actions_1, self.n_actions_2

        for s1_size in range(1, n1 + 1):
            for s1_indices in combinations(range(n1), s1_size):
                for s2_size in range(1, n2 + 1):
                    for s2_indices in combinations(range(n2), s2_size):
                        eq = self._solve_support(
                            list(s1_indices), list(s2_indices)
                        )
                        if eq is not None:
                            sigma1, sigma2 = eq
                            # Check for duplicates
                            is_dup = False
                            for e1, e2 in equilibria:
                                if (np.allclose(e1, sigma1, atol=1e-5) and
                                        np.allclose(e2, sigma2, atol=1e-5)):
                                    is_dup = True
                                    break
                            if not is_dup:
                                equilibria.append((sigma1, sigma2))

        return equilibria

    def _solve_support(self, s1_indices, s2_indices):
        """Attempt to solve for a mixed Nash on given supports.

        Parameters
        ----------
        s1_indices : list of int
            Support indices for player 1.
        s2_indices : list of int
            Support indices for player 2.

        Returns
        -------
        tuple or None
            (sigma1, sigma2) if a valid equilibrium exists; None otherwise.
        """
        n1, n2 = self.n_actions_1, self.n_actions_2
        k1, k2 = len(s1_indices), len(s2_indices)

        if k1 > k2 + 1 or k2 > k1 + 1:
            return None

        # Solve for player 2's strategy: for each s1_i in support,
        # payoff must be equal. We use the condition that on the support,
        # player 1 is indifferent.
        # Build system: for j in s2_indices, sum_j B[i][j] * sigma2[j] = v2
        # plus sum_j sigma2[j] = 1.
        if k2 == 1:
            sigma2 = np.zeros(n2)
            sigma2[s2_indices[0]] = 1.0
        else:
            # k2 equations for k2 variables (sigma2 on support)
            # k2-1 indifference equations + sum = 1
            M = np.zeros((k2, k2))
            rhs = np.zeros(k2)
            # Indifference: for i = s1_indices[0..k2-2] vs s1_indices[k2-1]
            ref_idx = s1_indices[min(k2 - 1, k1 - 1)]
            eq_count = 0
            for i in range(min(k2 - 1, k1)):
                if eq_count >= k2 - 1:
                    break
                row_idx = s1_indices[i]
                if row_idx == ref_idx and i > 0:
                    continue
                for j_idx, j in enumerate(s2_indices):
                    M[eq_count, j_idx] = (
                        self.A[row_idx, j] - self.A[ref_idx, j]
                    )
                rhs[eq_count] = 0.0
                eq_count += 1
            # If we still need equations, use best response indifference
            while eq_count < k2 - 1:
                M[eq_count, :] = 0
                M[eq_count, 0] = 1
                M[eq_count, 1] = -1
                rhs[eq_count] = 0
                eq_count += 1
            # Sum = 1
            M[k2 - 1, :] = 1.0
            rhs[k2 - 1] = 1.0

            try:
                sigma2_sub = np.linalg.solve(M, rhs)
                sigma2 = np.zeros(n2)
                for j_idx, j in enumerate(s2_indices):
                    sigma2[j] = sigma2_sub[j_idx]
            except np.linalg.LinAlgError:
                return None

        # Check non-negativity
        if np.any(sigma2 < -1e-8):
            return None
        sigma2 = np.maximum(sigma2, 0)
        sigma2 = sigma2 / sigma2.sum()

        # Solve for player 1's strategy similarly
        if k1 == 1:
            sigma1 = np.zeros(n1)
            sigma1[s1_indices[0]] = 1.0
        else:
            M = np.zeros((k1, k1))
            rhs = np.zeros(k1)
            ref_idx = s2_indices[min(k1 - 1, k2 - 1)]
            eq_count = 0
            for j in range(min(k1 - 1, k2)):
                if eq_count >= k1 - 1:
                    break
                col_idx = s2_indices[j]
                if col_idx == ref_idx and j > 0:
                    continue
                for i_idx, i in enumerate(s1_indices):
                    M[eq_count, i_idx] = (
                        self.B[i, col_idx] - self.B[i, ref_idx]
                    )
                rhs[eq_count] = 0.0
                eq_count += 1
            while eq_count < k1 - 1:
                M[eq_count, :] = 0
                M[eq_count, 0] = 1
                M[eq_count, 1] = -1
                rhs[eq_count] = 0
                eq_count += 1
            M[k1 - 1, :] = 1.0
            rhs[k1 - 1] = 1.0

            try:
                sigma1_sub = np.linalg.solve(M, rhs)
                sigma1 = np.zeros(n1)
                for i_idx, i in enumerate(s1_indices):
                    sigma1[i] = sigma1_sub[i_idx]
            except np.linalg.LinAlgError:
                return None

        if np.any(sigma1 < -1e-8):
            return None
        sigma1 = np.maximum(sigma1, 0)
        sigma1 = sigma1 / sigma1.sum()

        # Verify the solution
        if self.is_nash((sigma1, sigma2)):
            return (sigma1, sigma2)
        return None

    def summary(self):
        """Print a summary of the game."""
        print(f"Normal-Form Game: {self.n_actions_1} x {self.n_actions_2}")
        print(f"\n{self.player_names[0]}'s Payoff Matrix:")
        print(self.A)
        print(f"\n{self.player_names[1]}'s Payoff Matrix:")
        print(self.B)


def lemke_howson(A, B, initial_label=0):
    """Lemke-Howson algorithm for computing Nash equilibrium of bimatrix games.

    Implements the classic complementary pivoting algorithm for finding
    one Nash equilibrium of a bimatrix game.

    If the standard pivot fails (ray termination), tries the alternate
    initial label. If both fail, falls back to support enumeration.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Payoff matrix for player 1 (row player).
    B : ndarray, shape (m, n)
        Payoff matrix for player 2 (column player).
    initial_label : int, optional
        Initial label to drop (0 for label 1, 1 for label m+1).

    Returns
    -------
    tuple of ndarray
        (sigma1, sigma2) - mixed strategy Nash equilibrium.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape

    # Try both initial labels
    for init_label in [initial_label, (initial_label + 1) % (m + n)]:
        result = _lemke_howson_single(A, B, init_label)
        if result is not None:
            s1, s2 = result
            if is_nash((s1, s2), (A, B)):
                return result

    # Fallback: use support enumeration
    game = NormalFormGame(A, B)
    equilibria = game.find_nash_brute_force()
    if equilibria:
        return equilibria[0]

    # Ultimate fallback
    return np.ones(m) / m, np.ones(n) / n


def _lemke_howson_single(A, B, initial_label):
    """Single run of Lemke-Howson with a given initial label.

    Returns None if the algorithm hits a ray (no equilibrium found via
    this path).
    """
    m, n = A.shape

    # Ensure payoffs are positive for numerical stability
    min_val = min(A.min(), B.min())
    if min_val <= 0:
        shift = -min_val + 1.0
        A_shifted = A + shift
        B_shifted = B + shift
    else:
        A_shifted = A
        B_shifted = B

    total = m + n

    # Build the LCP tableau: [ -M | I | q ]
    # where z = [x; y], w = [u; v]
    # u_i = 1 - sum_j B[j,i] x_j  (i from 0..n-1)
    # v_j = 1 - sum_i A[j,i] y_i  (j from 0..m-1)
    #
    # The tableau variables are ordered: z_0..z_{total-1}, w_0..w_{total-1}
    # Initial basis: w_0..w_{total-1} (columns total..2*total-1)
    #
    # For row i (0..n-1, corresponding to u_i):
    #   u_i = 1 - sum_{k=0}^{m-1} B_shifted[k,i] z_k
    #   So -M[i, k] = -B_shifted[k,i] for k in 0..m-1
    #
    # For row n+j (corresponding to v_j, j in 0..m-1):
    #   v_j = 1 - sum_{k=0}^{n-1} A_shifted[j,k] z_{m+k}
    #   So -M[n+j, m+k] = -A_shifted[j,k] for k in 0..n-1

    tableau = np.zeros((total, 2 * total + 1))
    # -M entries (z columns)
    for i in range(n):
        for k in range(m):
            tableau[i, k] = -B_shifted[k, i]
    for j in range(m):
        for k in range(n):
            tableau[n + j, m + k] = -A_shifted[j, k]
    # Identity (w columns)
    for i in range(total):
        tableau[i, total + i] = 1.0
    # q (RHS)
    tableau[:, -1] = 1.0

    basis = list(range(total, 2 * total))  # all w basic initially
    entering = initial_label
    max_iter = 100 * total

    for _ in range(max_iter):
        col = tableau[:, entering]

        # Ratio test
        ratios = np.full(total, np.inf)
        for i in range(total):
            if col[i] > 1e-10:
                ratios[i] = tableau[i, -1] / col[i]

        leaving_row = np.argmin(ratios)
        if ratios[leaving_row] == np.inf:
            return None  # Ray termination

        # Pivot
        pivot_val = tableau[leaving_row, entering]
        tableau[leaving_row, :] /= pivot_val
        for i in range(total):
            if i != leaving_row:
                factor = tableau[i, entering]
                if abs(factor) > 1e-12:
                    tableau[i, :] -= factor * tableau[leaving_row, :]

        # Update basis
        leaving_var = basis[leaving_row]
        basis[leaving_row] = entering

        # Complement of leaving variable
        if leaving_var < total:
            complement = leaving_var + total
        else:
            complement = leaving_var - total

        if complement == initial_label:
            break

        entering = complement

    # Extract solution
    z_vals = np.zeros(total)
    for i, b in enumerate(basis):
        if b < total:
            z_vals[b] = max(0, tableau[i, -1])

    x_raw = z_vals[:m]
    y_raw = z_vals[m:]

    x_sum = x_raw.sum()
    y_sum = y_raw.sum()

    if x_sum > 1e-10:
        sigma1 = x_raw / x_sum
    else:
        sigma1 = np.ones(m) / m

    if y_sum > 1e-10:
        sigma2 = y_raw / y_sum
    else:
        sigma2 = np.ones(n) / n

    return sigma1, sigma2


def mixed_nash_2x2(A, B):
    """Closed-form solution for mixed Nash equilibrium of 2x2 games.

    Parameters
    ----------
    A : ndarray, shape (2, 2)
        Payoff matrix for player 1.
    B : ndarray, shape (2, 2)
        Payoff matrix for player 2.

    Returns
    -------
    list of tuple
        All Nash equilibria (pure and mixed), each as (sigma1, sigma2).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    equilibria = []

    # Check all 4 pure strategy profiles
    for i in range(2):
        for j in range(2):
            s1 = np.eye(2)[i]
            s2 = np.eye(2)[j]

            # Check if either player wants to deviate
            p1_alt = A[1 - i, j]
            p1_cur = A[i, j]
            p2_alt = B[i, 1 - j]
            p2_cur = B[i, j]

            if p1_alt <= p1_cur + 1e-10 and p2_alt <= p2_cur + 1e-10:
                equilibria.append((s1.copy(), s2.copy()))

    # Mixed equilibrium: player 1 indifferent between rows
    # sigma2[0] * A[0,0] + sigma2[1] * A[0,1] = sigma2[0] * A[1,0] + sigma2[1] * A[1,1]
    # sigma2[1] = 1 - sigma2[0]
    denom_p2 = A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1]
    if abs(denom_p2) > 1e-10:
        p2_0 = (A[1, 1] - A[0, 1]) / denom_p2
        p2_0 = np.clip(p2_0, 0, 1)

        denom_p1 = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
        if abs(denom_p1) > 1e-10:
            p1_0 = (B[1, 1] - B[0, 1]) / denom_p1
            p1_0 = np.clip(p1_0, 0, 1)

            if 0 < p1_0 < 1 and 0 < p2_0 < 1:
                s1 = np.array([p1_0, 1 - p1_0])
                s2 = np.array([p2_0, 1 - p2_0])
                equilibria.append((s1, s2))

    return equilibria


def correlated_equilibrium(payoffs):
    """Find correlated equilibria via linear programming.

    Computes the set of correlated equilibria using the incentive
    constraints as linear inequalities.

    Parameters
    ----------
    payoffs : tuple of ndarray
        (A, B) payoff matrices for player 1 and player 2.

    Returns
    -------
    ndarray
        A correlated equilibrium distribution over joint actions,
        shape (n_actions_1, n_actions_2).
    """
    A, B = payoffs
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n1, n2 = A.shape

    # Decision variables: p[i, j] for i=0..n1-1, j=0..n2-1
    # 1. sum_{i,j} p[i,j] = 1
    # 2. p[i,j] >= 0
    # 3. For each i, i', sum_j (A[i,j] - A[i',j]) * p[i,j] >= 0
    # 4. For each j, j', sum_i (B[i,j] - B[i,j']) * p[i,j] >= 0

    # We'll minimize 0 (any solution works) with constraints
    n_vars = n1 * n2

    # Objective: minimize 0
    c = np.zeros(n_vars)

    # Equality constraint: sum(p) = 1
    A_eq = np.ones((1, n_vars))
    b_eq = np.ones(1)

    # Inequality constraints: incentive constraints
    n_incentive_1 = n1 * (n1 - 1)  # for each i, i'
    n_incentive_2 = n2 * (n2 - 1)  # for each j, j'
    n_ineq = n_incentive_1 + n_incentive_2

    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    row = 0
    # Player 1 incentive constraints
    for i in range(n1):
        for i_prime in range(n1):
            if i == i_prime:
                continue
            for j in range(n2):
                A_ub[row, i * n2 + j] = A[i_prime, j] - A[i, j]
            b_ub[row] = 0.0
            row += 1

    # Player 2 incentive constraints
    for j in range(n2):
        for j_prime in range(n2):
            if j == j_prime:
                continue
            for i in range(n1):
                A_ub[row, i * n2 + j] = B[i, j_prime] - B[i, j]
            b_ub[row] = 0.0
            row += 1

    # Bounds: p >= 0
    bounds = [(0, None) for _ in range(n_vars)]

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        p = result.x.reshape(n1, n2)
        p = np.maximum(p, 0)
        p = p / p.sum()
        return p
    else:
        # Fallback: return uniform distribution
        return np.ones((n1, n2)) / (n1 * n2)


def is_nash(strategies, payoffs, tol=1e-6):
    """Verify whether a strategy profile is a Nash equilibrium.

    Parameters
    ----------
    strategies : tuple of ndarray
        Strategy profile (sigma1, sigma2).
    payoffs : tuple of ndarray
        (A, B) payoff matrices.
    tol : float
        Numerical tolerance.

    Returns
    -------
    bool
        True if the strategy profile is a Nash equilibrium.
    """
    A, B = payoffs
    sigma1, sigma2 = strategies

    p1_payoffs = A @ sigma2
    p1_best = np.max(p1_payoffs)
    for i in range(len(sigma1)):
        if sigma1[i] > tol and p1_payoffs[i] < p1_best - tol:
            return False

    p2_payoffs = sigma1 @ B
    p2_best = np.max(p2_payoffs)
    for j in range(len(sigma2)):
        if sigma2[j] > tol and p2_payoffs[j] < p2_best - tol:
            return False

    return True


def best_response(strategies, payoffs, player):
    """Compute the best response of a player to a strategy profile.

    Parameters
    ----------
    strategies : tuple of ndarray
        Current strategy profile (sigma1, sigma2).
    payoffs : tuple of ndarray
        (A, B) payoff matrices.
    player : int
        Player to compute best response for (1 or 2).

    Returns
    -------
    ndarray
        Indices of pure strategy best responses.
    """
    A, B = payoffs
    sigma1, sigma2 = strategies
    if player == 1:
        p = A @ sigma2
    else:
        p = sigma1 @ B
    max_val = np.max(p)
    return np.where(np.abs(p - max_val) < 1e-10)[0]


def fictitious_play(payoffs, n_iter=1000):
    """Fictitious play for approximating Nash equilibrium in large games.

    Parameters
    ----------
    payoffs : tuple of ndarray
        (A, B) payoff matrices.
    n_iter : int
        Number of iterations.

    Returns
    -------
    tuple
        (sigma1, sigma2, history) where sigma1, sigma2 are the average
        empirical strategy distributions, and history is a dict with
        per-iteration payoffs.
    """
    A, B = payoffs
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n1, n2 = A.shape

    # Counts of actions played
    count1 = np.zeros(n1)
    count2 = np.zeros(n2)

    # Initial pure strategies
    a1 = 0
    a2 = 0

    history = {'payoff1': [], 'payoff2': [], 'action1': [], 'action2': []}

    for t in range(1, n_iter + 1):
        count1[a1] += 1
        count2[a2] += 1

        # Update empirical distributions
        sigma1 = count1 / t
        sigma2 = count2 / t

        # Best response to opponent's empirical distribution
        p1_payoffs = A @ sigma2
        a1 = np.argmax(p1_payoffs)

        p2_payoffs = sigma1 @ B
        a2 = np.argmax(p2_payoffs)

        history['payoff1'].append(float(p1_payoffs[a1]))
        history['payoff2'].append(float(p2_payoffs[a2]))
        history['action1'].append(a1)
        history['action2'].append(a2)

    sigma1 = count1 / n_iter
    sigma2 = count2 / n_iter

    return sigma1, sigma2, history
