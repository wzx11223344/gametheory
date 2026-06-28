# GameTheory - Game Theory & Mechanism Design Library

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library for **Game Theory** and **Mechanism Design**, providing ready-to-use implementations of classic algorithms and models.

## Features

### Normal-Form Games
- Nash equilibrium computation (brute force, Lemke-Howson, 2x2 closed-form)
- Correlated equilibrium via linear programming
- Fictitious play for large games
- Best response and equilibrium verification

### Extensive-Form Games
- Backward induction for perfect-information games
- Subgame perfect equilibrium
- Classic examples: Entry deterrence, Centipede game

### Auctions & Mechanism Design
- First-price sealed-bid (FPSB) and second-price sealed-bid (SPSB/Vickrey)
- VCG mechanism
- Revenue equivalence Monte Carlo simulation
- English (ascending clock) auction simulator
- Optimal reserve price (Myerson)
- Bayesian Nash equilibrium bidding strategies

### Matching Markets
- Gale-Shapley deferred acceptance (stable marriage)
- School choice (student-proposing DA)
- Top Trading Cycles (TTC)
- Random Serial Dictatorship (RSD)
- Stability verification

### Bargaining Theory
- Rubinstein alternating-offers bargaining
- Nash bargaining solution (axiomatic)
- Kalai-Smorodinsky solution
- Egalitarian solution

### Evolutionary Game Theory
- Continuous and discrete replicator dynamics
- Evolutionarily Stable Strategy (ESS) detection
- Hawk-Dove and Prisoner's Dilemma as evolutionary games
- Phase diagram plotting

## Installation

```bash
pip install -e .
```

Or simply:

```bash
pip install numpy scipy
```

## Quick Start

```python
import numpy as np
from gametheory import NormalFormGame

# Prisoner's Dilemma
A = np.array([[-1, -3], [0, -2]])
B = np.array([[-1, 0], [-3, -2]])
game = NormalFormGame(A, B)
game.summary()
```

```python
from gametheory.matching import deferred_acceptance

men_prefs = [[0, 1], [1, 0]]
women_prefs = [[0, 1], [1, 0]]
matching = deferred_acceptance(men_prefs, women_prefs)
```

```python
from gametheory.auctions import revenue_equivalence_test
revenue_equivalence_test(n_simulations=10000)
```

## Run Examples

```bash
python examples/demo.py
```

## Modules

| Module | Description |
|--------|-------------|
| `gametheory.normal_form` | Normal-form games, Nash equilibrium algorithms |
| `gametheory.extensive_form` | Extensive-form games, backward induction, SPE |
| `gametheory.auctions` | Auction theory, VCG, revenue equivalence |
| `gametheory.matching` | Matching markets, Gale-Shapley, TTC |
| `gametheory.bargaining` | Bargaining solutions, Rubinstein, Nash |
| `gametheory.evolutionary` | Replicator dynamics, ESS, evolutionary games |

## License

MIT License - see [LICENSE](LICENSE) for details.
