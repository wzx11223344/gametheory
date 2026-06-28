#!/usr/bin/env python
"""
GameTheory Library - Comprehensive Demonstration
================================================

This script demonstrates all major modules of the GameTheory library:
1. Normal-form games (Nash equilibrium, correlated equilibrium, fictitious play)
2. Extensive-form games (backward induction, SPE)
3. Auctions (FPSB, SPSB, revenue equivalence, VCG, English auction)
4. Matching markets (Gale-Shapley, school choice, TTC, RSD)
5. Bargaining theory (Rubinstein, Nash, Kalai-Smorodinsky, egalitarian)
6. Evolutionary game theory (replicator dynamics, ESS, Hawk-Dove)

Usage:
    python demo.py
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for local import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gametheory.normal_form import (
    NormalFormGame, lemke_howson, mixed_nash_2x2,
    correlated_equilibrium, is_nash, best_response, fictitious_play
)
from gametheory.extensive_form import (
    ExtensiveFormGame, backward_induction,
    subgame_perfect_equilibrium, compute_SPE,
    entry_deterrence_game, centipede_game, print_strategy
)
from gametheory.auctions import (
    first_price_auction, second_price_auction, simulate_fpsb,
    revenue_equivalence_test, english_auction_simulator,
    vcg_mechanism, optimal_reserve_price, symmetric_ipv_bid
)
from gametheory.matching import (
    deferred_acceptance, school_choice, top_trading_cycles,
    is_stable, random_serial_dictatorship, generate_random_preferences
)
from gametheory.bargaining import (
    rubinstein_bargaining, nash_bargaining_solution,
    kalai_smorodinsky, egalitarian_solution
)
from gametheory.evolutionary import (
    replicator_dynamics, discrete_replicator, find_ESS,
    hawk_dove_game, prisoners_dilemma, plot_dynamics
)


def print_separator(title):
    """Print a formatted section separator."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================================
# 1. NORMAL-FORM GAMES
# ============================================================================

def demo_normal_form():
    """Demonstrate normal-form game algorithms."""
    print_separator("1. NORMAL-FORM GAMES")

    # --- Prisoner's Dilemma ---
    print("\n[1a] Prisoner's Dilemma")
    A_pd = np.array([[-1, -3], [0, -2]])  # Player 1: Cooperate=0, Defect=1
    B_pd = np.array([[-1, 0], [-3, -2]])  # Player 2: Cooperate=0, Defect=1

    game_pd = NormalFormGame(A_pd, B_pd, ["Prisoner 1", "Prisoner 2"])
    game_pd.summary()

    # Find Nash equilibria
    nash_list = game_pd.find_nash_brute_force()
    print("\nNash Equilibria found:")
    for idx, (s1, s2) in enumerate(nash_list):
        print(f"  Equilibrium {idx+1}: sigma1={s1}, sigma2={s2}")
        print(f"    Player 1 payoff: {s1 @ A_pd @ s2:.2f}")
        print(f"    Player 2 payoff: {s1 @ B_pd @ s2:.2f}")
        print(f"    Is Nash: {game_pd.is_nash((s1, s2))}")

    # Verify using 2x2 closed-form
    nash_2x2 = mixed_nash_2x2(A_pd, B_pd)
    print(f"\n2x2 closed-form solutions: {len(nash_2x2)} found")

    # --- Battle of the Sexes ---
    print("\n[1b] Battle of the Sexes")
    A_bos = np.array([[3, 0], [0, 2]])  # Opera=0, Football=1
    B_bos = np.array([[2, 0], [0, 3]])
    game_bos = NormalFormGame(A_bos, B_bos, ["Wife", "Husband"])
    game_bos.summary()

    nash_bos = game_bos.find_nash_brute_force()
    print("\nNash Equilibria:")
    for idx, (s1, s2) in enumerate(nash_bos):
        p1 = s1[0]
        p2 = s2[0]
        print(f"  Eq {idx+1}: P1 plays Opera with prob {p1:.3f}, "
              f"P2 plays Opera with prob {p2:.3f}")
        print(f"    P1 payoff: {s1 @ A_bos @ s2:.2f}, "
              f"P2 payoff: {s1 @ B_bos @ s2:.2f}")

    # Find correlated equilibrium
    print("\n[1c] Correlated Equilibrium (Battle of the Sexes)")
    ce = correlated_equilibrium((A_bos, B_bos))
    print("Correlated Equilibrium distribution:")
    print(f"  P(Opera, Opera) = {ce[0, 0]:.4f}")
    print(f"  P(Opera, Football) = {ce[0, 1]:.4f}")
    print(f"  P(Football, Opera) = {ce[1, 0]:.4f}")
    print(f"  P(Football, Football) = {ce[1, 1]:.4f}")

    # Compute expected payoffs under CE
    ce_p1 = np.sum(ce * A_bos)
    ce_p2 = np.sum(ce * B_bos)
    print(f"  Expected payoff P1: {ce_p1:.2f}")
    print(f"  Expected payoff P2: {ce_p2:.2f}")

    # --- Fictitious Play ---
    print("\n[1d] Fictitious Play (Matching Pennies)")
    A_mp = np.array([[1, -1], [-1, 1]])
    B_mp = np.array([[-1, 1], [1, -1]])
    sigma1_fp, sigma2_fp, hist = fictitious_play((A_mp, B_mp), n_iter=10000)
    print(f"  Converged to: sigma1 = {sigma1_fp}, sigma2 = {sigma2_fp}")
    print(f"  (Expected: uniform [0.5, 0.5] for Matching Pennies)")

    # --- Lemke-Howson ---
    print("\n[1e] Lemke-Howson Algorithm")
    try:
        s1_lh, s2_lh = lemke_howson(A_bos, B_bos)
        print(f"  Lemke-Howson result: sigma1 = {np.round(s1_lh, 4)}, "
              f"sigma2 = {np.round(s2_lh, 4)}")
        print(f"  Is Nash: {is_nash((s1_lh, s2_lh), (A_bos, B_bos))}")
    except Exception as e:
        print(f"  Lemke-Howson failed: {e}")


# ============================================================================
# 2. EXTENSIVE-FORM GAMES
# ============================================================================

def demo_extensive_form():
    """Demonstrate extensive-form game algorithms."""
    print_separator("2. EXTENSIVE-FORM GAMES")

    # --- Entry Deterrence ---
    print("\n[2a] Entry Deterrence Game")
    game_ed = entry_deterrence_game()
    payoffs, strategy = compute_SPE(game_ed)
    print(f"  SPE Payoffs: {payoffs}")
    print_strategy(game_ed, strategy)

    # --- Centipede Game ---
    print("\n[2b] Centipede Game (4 stages)")
    game_cent = centipede_game(n_stages=4, payoff_increment=1)
    payoffs_c, strategy_c = backward_induction(game_cent)
    print(f"  SPE Payoffs: {payoffs_c}")
    print(f"  (Unique SPE: Player 1 Takes immediately at stage 1)")
    print(f"  Centipede paradox: backward induction selects the "
          f"inefficient outcome (Take, Take, ...)")
    print(f"  Total possible surplus if both cooperate: "
          f"{2 * 4 + 2}")


# ============================================================================
# 3. AUCTIONS
# ============================================================================

def demo_auctions():
    """Demonstrate auction theory algorithms."""
    print_separator("3. AUCTIONS & MECHANISM DESIGN")

    # --- FPSB vs SPSB single run ---
    print("\n[3a] Single Auction Comparison")
    np.random.seed(42)
    values = np.array([0.7, 0.3, 0.5])

    fpsb_res = first_price_auction(3, values)
    spsb_res = second_price_auction(3, values)

    print(f"  Values: {values}")
    print(f"  FPSB: winner={fpsb_res['winner']}, revenue={fpsb_res['revenue']:.3f}")
    print(f"  SPSB: winner={spsb_res['winner']}, revenue={spsb_res['revenue']:.3f}")

    # --- Revenue Equivalence Test ---
    print("\n[3b] Revenue Equivalence Test (10,000 simulations)")
    ret_result = revenue_equivalence_test(n_simulations=10000)
    print(f"  FPSB avg revenue: {ret_result['fpsb_revenue']:.4f}")
    print(f"  SPSB avg revenue: {ret_result['spsb_revenue']:.4f}")
    print(f"  Difference: {ret_result['difference']:.6f}")
    print(f"  FPSB efficiency: {ret_result['fpsb_efficiency']:.4f}")
    print(f"  SPSB efficiency: {ret_result['spsb_efficiency']:.4f}")
    print(f"  Revenue Equivalence confirmed: {ret_result['confirmed']}")

    # --- English Auction ---
    print("\n[3c] English (Ascending Clock) Auction")
    eng_result = english_auction_simulator(n_bidders=5)
    print(f"  Values: {np.round(eng_result['values'], 3)}")
    print(f"  Winner: Bidder {eng_result['winner']}")
    print(f"  Final price (revenue): {eng_result['revenue']:.3f}")
    print(f"  Rounds: {eng_result['rounds']}")

    # --- VCG Mechanism ---
    print("\n[3d] VCG Mechanism (3 bidders, 2 items)")
    bidder_vals = np.array([
        [10, 5],   # Bidder 1 values
        [8, 3],    # Bidder 2 values
        [6, 7],    # Bidder 3 values
    ])
    vcg_res = vcg_mechanism(bidder_vals, items=2)
    print(f"  Bidder values:\n{bidder_vals}")
    print(f"  Winners: {vcg_res['allocation']}")
    print(f"  Payments: {np.round(vcg_res['payments'], 2)}")
    print(f"  Social welfare: {vcg_res['social_welfare']}")

    # --- Optimal Reserve Price ---
    print("\n[3e] Myerson Optimal Reserve Price (Uniform[0,1])")
    def F_uniform(x):
        return np.clip(x, 0, 1)

    def f_uniform(x):
        x_arr = np.atleast_1d(x)
        result = np.where((x_arr >= 0) & (x_arr <= 1), 1.0, 0.0)
        return float(result[0]) if np.isscalar(x) else result

    opt_r = optimal_reserve_price((F_uniform, f_uniform), n_bidders=2)
    print(f"  Optimal reserve price: {opt_r:.4f} (expected: 0.5 for Uniform)")

    # --- Bayesian Nash Equilibrium Bid ---
    print("\n[3f] Bayesian Nash Equilibrium Bidding (Uniform[0,1])")
    for v_test in [0.3, 0.5, 0.7, 0.9]:
        bid = symmetric_ipv_bid(v_test, n=3, F=F_uniform, f=f_uniform)
        expected = (2/3) * v_test  # Uniform: (n-1)/n * v
        print(f"  v={v_test:.1f}: b*(v)={bid:.4f} (expected: {expected:.4f})")


# ============================================================================
# 4. MATCHING MARKETS
# ============================================================================

def demo_matching():
    """Demonstrate matching market algorithms."""
    print_separator("4. MATCHING MARKETS")

    # --- Gale-Shapley (Stable Marriage) ---
    print("\n[4a] Gale-Shapley Deferred Acceptance (Men-Proposing)")
    men_prefs = [
        [0, 1, 2],  # Man 0 prefers W0 > W1 > W2
        [1, 0, 2],  # Man 1 prefers W1 > W0 > W2
        [0, 1, 2],  # Man 2 prefers W0 > W1 > W2
    ]
    women_prefs = [
        [1, 0, 2],  # Woman 0 prefers M1 > M0 > M2
        [2, 0, 1],  # Woman 1 prefers M2 > M0 > M1
        [0, 1, 2],  # Woman 2 prefers M0 > M1 > M2
    ]

    matching_m = deferred_acceptance(men_prefs, women_prefs, proposing_side='men')
    print(f"  Men-Proposing Matching: {matching_m}")
    for m, w in matching_m.items():
        if w is not None:
            print(f"    Man {m} <-> Woman {w}")
        else:
            print(f"    Man {m} is unmatched")

    # Women-proposing
    matching_w = deferred_acceptance(men_prefs, women_prefs, proposing_side='women')
    print(f"\n  Women-Proposing Matching: {matching_w}")

    # Stability check
    stable, blocking = is_stable(matching_m, (men_prefs, women_prefs), side='men')
    print(f"\n  Men-proposing matching is stable: {stable}")
    if blocking:
        print(f"  Blocking pairs: {blocking}")

    # --- School Choice ---
    print("\n[4b] School Choice (Student-Proposing DA)")
    students = [
        [0, 1, 2],  # Student 0: S0 > S1 > S2
        [0, 2, 1],  # Student 1: S0 > S2 > S1
        [1, 0, 2],  # Student 2: S1 > S0 > S2
        [2, 1, 0],  # Student 3: S2 > S1 > S0
    ]
    school_priorities = [
        [3, 0, 1, 2],  # School 0 priority
        [0, 2, 1, 3],  # School 1 priority
        [1, 2, 3, 0],  # School 2 priority
    ]
    capacities = [2, 1, 1]

    student_assign, school_assign = school_choice(
        students, school_priorities, capacities
    )
    print(f"  Capacities: {capacities}")
    print(f"  Student assignments: {student_assign}")
    print(f"  School rosters: {school_assign}")

    # --- Top Trading Cycles ---
    print("\n[4c] Top Trading Cycles (TTC)")
    ttc_assign = top_trading_cycles(
        students, school_priorities, capacities, school_priorities
    )
    print(f"  TTC assignments: {ttc_assign}")

    # --- Random Serial Dictatorship ---
    print("\n[4d] Random Serial Dictatorship")
    preferences = generate_random_preferences(5, 3, seed=42)
    rsd_alloc = random_serial_dictatorship(5, 3, preferences)
    print(f"  Preferences: {preferences}")
    print(f"  RSD allocation: {rsd_alloc}")


# ============================================================================
# 5. BARGAINING THEORY
# ============================================================================

def demo_bargaining():
    """Demonstrate bargaining theory algorithms."""
    print_separator("5. BARGAINING THEORY")

    # --- Rubinstein Bargaining ---
    print("\n[5a] Rubinstein Alternating-Offers Bargaining")
    v = 100.0

    # Equal patience
    s1, s2 = rubinstein_bargaining(v, delta1=0.9, delta2=0.9)
    print(f"  Equal patience (d1=0.9, d2=0.9): ({s1:.2f}, {s2:.2f})")

    # Impatient player 1
    s1, s2 = rubinstein_bargaining(v, delta1=0.5, delta2=0.9)
    print(f"  Impatient P1 (d1=0.5, d2=0.9): ({s1:.2f}, {s2:.2f})")

    # Very patient player 1
    s1, s2 = rubinstein_bargaining(v, delta1=0.99, delta2=0.5)
    print(f"  Patient P1 (d1=0.99, d2=0.5): ({s1:.2f}, {s2:.2f})")

    # Finite horizon
    s1, s2 = rubinstein_bargaining(v, delta1=0.8, delta2=0.8, T=3)
    print(f"  Finite T=3 (d1=0.8, d2=0.8): ({s1:.2f}, {s2:.2f})")
    s1, s2 = rubinstein_bargaining(v, delta1=0.8, delta2=0.8, T=10)
    print(f"  Finite T=10 (d1=0.8, d2=0.8): ({s1:.2f}, {s2:.2f})")

    # --- Nash Bargaining Solution ---
    print("\n[5b] Nash Bargaining Solution")
    # Feasible set: u1 + u2 <= 10, u1 >= 0, u2 >= 0
    feasible = np.array([
        [10, 0], [9, 1], [8, 2], [7, 3], [6, 4],
        [5, 5], [4, 6], [3, 7], [2, 8], [1, 9], [0, 10]
    ])
    nash_sol = nash_bargaining_solution(feasible, disagreement=(0, 0))
    print(f"  Feasible set: u1 + u2 <= 10")
    print(f"  Nash solution: ({nash_sol[0]:.2f}, {nash_sol[1]:.2f})")
    print(f"  (Expected: (5, 5) for symmetric case)")

    # With asymmetric disagreement
    nash_sol2 = nash_bargaining_solution(feasible, disagreement=(2, 1))
    print(f"  With d=(2,1): ({nash_sol2[0]:.2f}, {nash_sol2[1]:.2f})")

    # --- Kalai-Smorodinsky ---
    print("\n[5c] Kalai-Smorodinsky Solution")
    ks_sol = kalai_smorodinsky(feasible, ideal_point=(10, 10))
    print(f"  KS solution (ideal=(10,10)): ({ks_sol[0]:.2f}, {ks_sol[1]:.2f})")

    ks_sol2 = kalai_smorodinsky(feasible, ideal_point=(10, 5))
    print(f"  KS solution (ideal=(10,5)): ({ks_sol2[0]:.2f}, {ks_sol2[1]:.2f})")

    # --- Egalitarian ---
    print("\n[5d] Egalitarian Solution")
    egal_sol = egalitarian_solution(feasible)
    print(f"  Egalitarian solution: ({egal_sol[0]:.2f}, {egal_sol[1]:.2f})")


# ============================================================================
# 6. EVOLUTIONARY GAME THEORY
# ============================================================================

def demo_evolutionary():
    """Demonstrate evolutionary game theory algorithms."""
    print_separator("6. EVOLUTIONARY GAME THEORY")

    # --- Hawk-Dove ---
    print("\n[6a] Hawk-Dove Game (v=2, c=3)")
    A_hd, mixed_ess, pure_ess = hawk_dove_game(v=2.0, c=3.0)
    print(f"  Payoff matrix:\n{A_hd}")
    print(f"  Mixed ESS: play Hawk with prob {mixed_ess[0]:.3f} "
          f"(expected: v/c = 2/3 = 0.667)")
    print(f"  Pure ESS found: {len(pure_ess)}")

    for idx, ess in enumerate(pure_ess):
        print(f"    ESS {idx+1}: {ess}")

    # Replicator dynamics for Hawk-Dove
    x0_list = [
        np.array([0.9, 0.1]),  # Mostly Hawks
        np.array([0.3, 0.7]),  # Mostly Doves
        np.array([0.5, 0.5]),  # Equal mix
    ]
    trajectories = []
    for x0 in x0_list:
        t, x = replicator_dynamics(A_hd, x0, T=20, dt=0.01)
        trajectories.append((t, x))
        final_x = x[-1]
        print(f"  From x0={x0}: converges to {np.round(final_x, 4)}")

    # --- Prisoner's Dilemma (Evolutionary) ---
    print("\n[6b] Prisoner's Dilemma as Evolutionary Game (b=3, c=1)")
    A_pd_evo, ess_pd = prisoners_dilemma(b=3.0, c=1.0)
    print(f"  Payoff matrix:\n{A_pd_evo}")
    print(f"  ESS: {ess_pd}")
    for idx, ess in enumerate(ess_pd):
        strategy_name = "Cooperate" if ess[0] > 0.5 else "Defect"
        print(f"    ESS {idx+1}: {ess} -> {strategy_name}")

    # Replicator dynamics for PD
    x0_pd_list = [
        np.array([0.8, 0.2]),  # Mostly Cooperators
        np.array([0.2, 0.8]),  # Mostly Defectors
        np.array([0.5, 0.5]),  # Equal mix
    ]
    for x0 in x0_pd_list:
        t, x = replicator_dynamics(A_pd_evo, x0, T=10, dt=0.01)
        final_x = x[-1]
        print(f"  From x0={x0}: converges to {np.round(final_x, 4)}")

    # --- ESS Detection ---
    print("\n[6c] ESS Detection (General 3x3 Example)")
    A_3x3 = np.array([
        [2, 1, 0],
        [1, 2, 1],
        [0, 1, 2],
    ])
    ess_list = find_ESS(A_3x3)
    print(f"  Payoff matrix:\n{A_3x3}")
    print(f"  ESS found: {len(ess_list)}")
    for idx, ess in enumerate(ess_list):
        print(f"    ESS {idx+1}: {np.round(ess, 4)}")

    # --- Discrete Replicator ---
    print("\n[6d] Discrete Replicator Dynamics")
    x_disc = discrete_replicator(A_hd, np.array([0.8, 0.2]), n_generations=50)
    print(f"  Hawk-Dove discrete: initial [0.8, 0.2] -> final {np.round(x_disc[-1], 4)}")

    # --- Plot Dynamics ---
    print("\n[6e] Generating Phase Diagram...")
    try:
        fig = plot_dynamics(
            trajectories,
            labels=["Mostly Hawks", "Mostly Doves", "Equal Mix"],
            title="Hawk-Dove Replicator Dynamics (v=2, c=3)",
            save_path=os.path.join(os.path.dirname(__file__),
                                   "hawk_dove_dynamics.png")
        )
        print(f"  Phase diagram saved to: examples/hawk_dove_dynamics.png")
        plt.close(fig)
    except Exception as e:
        print(f"  Plotting failed (non-critical): {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  GameTheory Library - Comprehensive Demonstration")
    print("=" * 70)

    demo_normal_form()
    demo_extensive_form()
    demo_auctions()
    demo_matching()
    demo_bargaining()
    demo_evolutionary()

    print_separator("DEMONSTRATION COMPLETE")
    print("\nAll modules have been successfully demonstrated.")
    print("The GameTheory library is ready for use.\n")


if __name__ == "__main__":
    main()
