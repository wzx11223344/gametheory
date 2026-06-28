"""
GameTheory - Game Theory & Mechanism Design Library
====================================================

A comprehensive Python library for Game Theory and Mechanism Design.
Provides implementations of classic algorithms across normal-form games,
extensive-form games, auctions, matching markets, bargaining theory,
and evolutionary game theory.

Modules
-------
normal_form    : Normal-form games, Nash equilibrium algorithms
extensive_form : Extensive-form games, backward induction, SPE
auctions       : Auction theory, VCG mechanism, revenue equivalence
matching       : Matching markets, Gale-Shapley, TTC
bargaining     : Bargaining solutions, Rubinstein, Nash, Kalai-Smorodinsky
evolutionary   : Replicator dynamics, ESS, evolutionary games
"""

__version__ = "0.1.0"
__author__ = "GameTheory Contributors"
__all__ = [
    "normal_form",
    "extensive_form",
    "auctions",
    "matching",
    "bargaining",
    "evolutionary",
]

from gametheory.normal_form import NormalFormGame
from gametheory.extensive_form import ExtensiveFormGame
