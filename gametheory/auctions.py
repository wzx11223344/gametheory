"""
Auctions & Mechanism Design Module
==================================

Provides implementations of classic auction formats and mechanism design
algorithms, including first-price, second-price (Vickrey), VCG, English
auctions, and revenue equivalence analysis.

Algorithms implemented:
- First-price sealed-bid auction
- Second-price sealed-bid (Vickrey) auction
- Monte Carlo simulation of FPSB vs SPSB revenue equivalence
- English (ascending clock) auction simulator
- VCG mechanism
- Optimal reserve price (Myerson)
- Bayesian Nash equilibrium bidding strategies
"""

import numpy as np


def first_price_auction(n_bidders, values, reserve_price=None):
    """First-price sealed-bid auction.

    Each bidder submits a bid. The winner pays their own bid.

    Parameters
    ----------
    n_bidders : int
        Number of bidders.
    values : array-like or callable
        Private values for each bidder, or a callable that generates
        n_bidders values when called.
    reserve_price : float, optional
        Reserve price below which the item is not sold.

    Returns
    -------
    dict
        Auction result with keys: 'winner', 'winning_bid', 'revenue',
        'bids', 'values', 'efficient'.
    """
    if callable(values):
        v = np.asarray([values() for _ in range(n_bidders)])
    else:
        v = np.asarray(values)
        assert len(v) == n_bidders

    # In equilibrium, bidders shade their bids: b_i = (n-1)/n * v_i
    bids = (n_bidders - 1) / n_bidders * v

    # Apply reserve price
    if reserve_price is not None:
        valid = bids >= reserve_price
        if not np.any(valid):
            return {
                'winner': None,
                'winning_bid': 0,
                'revenue': 0,
                'bids': bids,
                'values': v,
                'efficient': False
            }
        bids = np.where(valid, bids, 0)

    winner = np.argmax(bids)
    winning_bid = bids[winner]

    # Check if reserve price met
    if reserve_price is not None and winning_bid < reserve_price:
        winner = None
        winning_bid = 0
        revenue = 0
        efficient = False
    else:
        revenue = winning_bid
        efficient = (winner == np.argmax(v))

    return {
        'winner': int(winner) if winner is not None else None,
        'winning_bid': float(winning_bid),
        'revenue': float(revenue),
        'bids': bids,
        'values': v,
        'efficient': efficient
    }


def second_price_auction(n_bidders, values, reserve_price=None):
    """Second-price sealed-bid (Vickrey) auction.

    Each bidder submits a bid. The winner pays the second-highest bid.
    Dominant strategy: bid truthfully (b_i = v_i).

    Parameters
    ----------
    n_bidders : int
        Number of bidders.
    values : array-like or callable
        Private values for each bidder.
    reserve_price : float, optional
        Reserve price.

    Returns
    -------
    dict
        Auction result with keys: 'winner', 'winning_bid', 'second_price',
        'revenue', 'bids', 'values', 'efficient'.
    """
    if callable(values):
        v = np.asarray([values() for _ in range(n_bidders)])
    else:
        v = np.asarray(values)
        assert len(v) == n_bidders

    # Dominant strategy: bid truthfully
    bids = v.copy()

    # Sort bids in descending order
    sorted_indices = np.argsort(bids)[::-1]
    highest_idx = sorted_indices[0]
    highest_bid = bids[highest_idx]

    if n_bidders >= 2:
        second_highest_bid = bids[sorted_indices[1]]
    else:
        second_highest_bid = 0

    winner = highest_idx
    revenue = second_highest_bid

    if reserve_price is not None and highest_bid < reserve_price:
        winner = None
        revenue = 0
        efficient = False
    else:
        revenue = max(second_highest_bid, reserve_price or 0)
        efficient = (winner == np.argmax(v))

    return {
        'winner': int(winner) if winner is not None else None,
        'winning_bid': float(highest_bid),
        'second_price': float(revenue),
        'revenue': float(revenue),
        'bids': bids,
        'values': v,
        'efficient': efficient
    }


def simulate_fpsb(n_bidders=3, n_simulations=10000,
                  value_distribution=None, reserve_price=None):
    """Monte Carlo simulation of first-price sealed-bid auction.

    Confirms revenue equivalence with second-price by computing
    average revenue over many random value draws.

    Parameters
    ----------
    n_bidders : int
        Number of bidders per auction.
    n_simulations : int
        Number of auctions to simulate.
    value_distribution : callable, optional
        Function that returns a random value. Default: Uniform[0,1].
    reserve_price : float, optional
        Reserve price.

    Returns
    -------
    dict
        Simulation results: 'avg_revenue', 'std_revenue', 'revenues',
        'efficiency_rate', 'avg_bids'.
    """
    if value_distribution is None:
        value_distribution = np.random.rand

    revenues = np.zeros(n_simulations)
    efficient_count = 0
    all_bids = []

    for i in range(n_simulations):
        result = first_price_auction(
            n_bidders, value_distribution, reserve_price
        )
        revenues[i] = result['revenue']
        if result['efficient']:
            efficient_count += 1
        all_bids.append(np.mean(result['bids']))

    return {
        'avg_revenue': float(np.mean(revenues)),
        'std_revenue': float(np.std(revenues)),
        'revenues': revenues,
        'efficiency_rate': efficient_count / n_simulations,
        'avg_bids': float(np.mean(all_bids))
    }


def revenue_equivalence_test(n_simulations=10000):
    """Compare revenue of FPSB vs SPSB auctions.

    Simulates both auction formats with the same value draws and
    compares average revenue to verify the Revenue Equivalence Theorem.

    Parameters
    ----------
    n_simulations : int
        Number of auctions to simulate.

    Returns
    -------
    dict
        Results for both formats: 'fpsb_revenue', 'spsb_revenue',
        'difference', 'confirmed'.
    """
    fpsb_revenues = np.zeros(n_simulations)
    spsb_revenues = np.zeros(n_simulations)
    fpsb_eff = 0
    spsb_eff = 0

    for i in range(n_simulations):
        values = np.random.rand(3)

        fpsb_res = first_price_auction(3, values)
        spsb_res = second_price_auction(3, values)

        fpsb_revenues[i] = fpsb_res['revenue']
        spsb_revenues[i] = spsb_res['revenue']
        if fpsb_res['efficient']:
            fpsb_eff += 1
        if spsb_res['efficient']:
            spsb_eff += 1

    diff = np.mean(fpsb_revenues) - np.mean(spsb_revenues)

    return {
        'fpsb_revenue': float(np.mean(fpsb_revenues)),
        'spsb_revenue': float(np.mean(spsb_revenues)),
        'fpsb_std': float(np.std(fpsb_revenues)),
        'spsb_std': float(np.std(spsb_revenues)),
        'difference': float(diff),
        'confirmed': abs(diff) < 0.02,
        'fpsb_efficiency': fpsb_eff / n_simulations,
        'spsb_efficiency': spsb_eff / n_simulations
    }


def english_auction_simulator(n_bidders=5, value_distribution=None,
                              start_price=0.0, increment=0.01):
    """Simulate an English (ascending clock) auction.

    Price increases incrementally. Bidders drop out when price exceeds
    their private value. Last remaining bidder wins at the price where
    the second-to-last bidder dropped out.

    Parameters
    ----------
    n_bidders : int
        Number of bidders.
    value_distribution : callable, optional
        Function generating private values. Default: Uniform[0,1].
    start_price : float
        Starting price.
    increment : float
        Price increment per round.

    Returns
    -------
    dict
        Auction result with 'winner', 'final_price', 'revenue',
        'values', 'drop_out_prices', 'rounds'.
    """
    if value_distribution is None:
        value_distribution = np.random.rand

    values = np.asarray([value_distribution() for _ in range(n_bidders)])
    active = np.ones(n_bidders, dtype=bool)
    drop_out_prices = np.full(n_bidders, np.nan)
    price = start_price
    rounds = 0

    while np.sum(active) > 1:
        rounds += 1
        price += increment
        # Bidders drop out if price exceeds their value
        dropping = active & (price > values)
        if np.any(dropping):
            drop_out_prices[dropping] = price
            active[dropping] = False

    winner = np.argmax(values)
    final_price = price

    # The winner pays the price at which the second-to-last bidder dropped
    remaining_values = values[drop_out_prices > 0]
    if len(remaining_values) >= 2:
        second_highest_price = np.sort(
            drop_out_prices[~np.isnan(drop_out_prices)]
        )[-1]
        # In English auction, winner pays second-highest value
        sorted_vals = np.sort(values)[::-1]
        if len(sorted_vals) >= 2:
            final_price = sorted_vals[1]
        else:
            final_price = max(drop_out_prices[~np.isnan(drop_out_prices)],
                              default=start_price)

    return {
        'winner': int(winner),
        'final_price': float(final_price),
        'revenue': float(final_price),
        'values': values,
        'drop_out_prices': drop_out_prices,
        'rounds': rounds
    }


def vcg_mechanism(bidders_values, items=1):
    """Vickrey-Clarke-Groves mechanism for efficient allocation.

    Allocates items efficiently and charges each bidder the externality
    they impose on others.

    Parameters
    ----------
    bidders_values : list of ndarray
        Each element is an array of a bidder's values for each item.
        Shape: (n_bidders, n_items) or list of n_item-length arrays.
    items : int
        Number of identical items to allocate (default 1).

    Returns
    -------
    dict
        'allocation': which bidder(s) get items,
        'payments': payment for each bidder,
        'social_welfare': total social welfare.
    """
    values = np.asarray(bidders_values)
    n_bidders, n_items = values.shape
    items = min(items, n_items, n_bidders)

    # Efficient allocation: give each item to highest-valuing bidder
    # For k identical items, choose k bidders with highest values
    # Simple case: each bidder wants at most 1 item
    # Find efficient allocation
    bidder_item_values = values[:, 0] if n_items == 1 else values.max(axis=1)
    sorted_indices = np.argsort(bidder_item_values)[::-1]
    winners = sorted_indices[:items]

    # Compute payments (VCG)
    # Payment = value of efficient alloc without bidder - (social welfare with bidder - bidder's value)
    payments = np.zeros(n_bidders)

    for i in range(n_bidders):
        # Social welfare without bidder i
        others = [j for j in range(n_bidders) if j != i]
        other_values = bidder_item_values[others]
        other_sorted = np.sort(other_values)[::-1]
        sw_without_i = np.sum(other_sorted[:items])

        # Social welfare with bidder i minus bidder i's value
        if i in winners:
            sw_with_i = np.sum(bidder_item_values[winners])
            sw_excluding_i = sw_with_i - bidder_item_values[i]
            payments[i] = max(0, sw_without_i - sw_excluding_i)
        else:
            payments[i] = 0

    social_welfare = np.sum(bidder_item_values[winners])

    return {
        'allocation': [int(w) for w in winners],
        'payments': payments,
        'social_welfare': float(social_welfare)
    }


def optimal_reserve_price(value_distribution, n_bidders=2):
    """Compute Myerson's optimal reserve price.

    For IPV model with distribution F, the optimal reserve price r
    satisfies: r - (1-F(r))/f(r) = 0 for the seller's own value.

    Parameters
    ----------
    value_distribution : tuple
        (F, f) where F is the CDF and f is the PDF as callables.
    n_bidders : int
        Number of bidders (for informational purposes).

    Returns
    -------
    float
        Optimal reserve price.
    """
    F, f = value_distribution

    # Solve r - (1-F(r))/f(r) = 0 via simple search
    best_r = 0.0
    best_val = -np.inf

    for r in np.linspace(0.01, 0.99, 200):
        val = r - (1.0 - F(r)) / max(f(r), 1e-10)
        if abs(val) < abs(best_val):
            best_val = val
            best_r = r

    return best_r


def symmetric_ipv_bid(value, n, F, f):
    """Bayesian Nash equilibrium bid function for symmetric IPV FPSB.

    In the symmetric independent private values model with distribution F
    (CDF) and f (PDF), the equilibrium bid is:

        b(v) = v - integral_0^v [F(x)^(n-1) / F(v)^(n-1)] dx

    Parameters
    ----------
    value : float
        Bidder's private valuation.
    n : int
        Number of bidders.
    F : callable
        Cumulative distribution function of values.
    f : callable
        Probability density function of values.

    Returns
    -------
    float
        Equilibrium bid amount.
    """
    # For uniform [0,1]: b(v) = (n-1)/n * v
    # General formula: b(v) = E[max_{j!=i} v_j | v_j <= v for all j != i]
    # b(v) = integral_0^v x * (n-1) * F(x)^(n-2) * f(x) dx / F(v)^(n-1)
    value = np.asarray(value, dtype=float)
    scalar = value.ndim == 0
    value = np.atleast_1d(value)
    result = np.zeros_like(value)

    for idx, v in enumerate(value):
        Fv = F(v)
        if Fv < 1e-12:
            result[idx] = 0.0
            continue

        # Numerical integration
        xs = np.linspace(0, v, 200)
        dx = xs[1] - xs[0]
        Fxs = np.array([F(x) for x in xs])
        fxs = np.array([f(x) for x in xs])
        integrand = xs * (n - 1) * np.power(np.maximum(Fxs, 1e-12), n - 2) * fxs
        expected = np.trapz(integrand, xs) / (Fv ** (n - 1))
        result[idx] = expected

    return float(result[0]) if scalar else result
