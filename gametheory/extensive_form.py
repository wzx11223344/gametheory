"""
Extensive-Form Game Module
==========================

Provides classes and algorithms for extensive-form games, including
backward induction, subgame perfect equilibrium, and classic examples.

Algorithms implemented:
- Backward induction for perfect-information games
- Subgame perfect equilibrium (SPE)
- Classic examples: Entry deterrence, Centipede game
"""

import numpy as np
from collections import defaultdict


class ExtensiveFormGame:
    """Extensive-form game with perfect information, represented as a tree.

    Each node is identified by a unique integer. The root is node 0.

    Parameters
    ----------
    n_players : int
        Number of players (>= 1).

    Attributes
    ----------
    n_players : int
        Number of players.
    children : dict
        Mapping from node_id to list of child node_ids.
    player : dict
        Mapping from node_id to player who moves at that node.
    payoffs : dict
        Mapping from terminal node_id to list of payoffs.
    actions : dict
        Mapping from (parent_node, child_node) to action label.
    parent : dict
        Mapping from child_node to parent_node.
    """

    def __init__(self, n_players=2):
        self.n_players = n_players
        self.children = defaultdict(list)
        self.player = {}
        self.payoffs = {}
        self.actions = {}
        self.parent = {}
        self._next_node = 1  # node 0 is root

    def add_decision_node(self, parent, player, n_actions, action_labels=None):
        """Add a decision node with multiple children.

        Sets the player who moves at the *parent* node. Creates n_actions
        child nodes that are the next states in the game.

        Parameters
        ----------
        parent : int
            Parent node ID (this is the decision node).
        player : int
            Player who moves at the parent node (0-indexed).
        n_actions : int
            Number of actions / child states to create.
        action_labels : list of str, optional
            Labels for each action (edge from parent to child).

        Returns
        -------
        list of int
            IDs of newly created child nodes.
        """
        if action_labels is None:
            action_labels = [f"a{i}" for i in range(n_actions)]

        # The parent is the decision node -- set the player there
        self.player[parent] = player

        new_nodes = []
        for i in range(n_actions):
            child = self._next_node
            self._next_node += 1
            self.children[parent].append(child)
            self.parent[child] = parent
            self.actions[(parent, child)] = action_labels[i]
            new_nodes.append(child)

        return new_nodes

    def add_terminal_node(self, parent, payoffs, action_label="term"):
        """Add a terminal (leaf) node with payoffs.

        Parameters
        ----------
        parent : int
            Parent node ID.
        payoffs : list of float
            Payoffs for each player at this terminal node.
        action_label : str, optional
            Label for the action leading to this node.

        Returns
        -------
        int
            ID of the newly created terminal node.
        """
        child = self._next_node
        self._next_node += 1
        self.children[parent].append(child)
        self.payoffs[child] = list(payoffs)
        self.parent[child] = parent
        self.actions[(parent, child)] = action_label
        return child

    def is_terminal(self, node):
        """Check if node is a terminal node.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        bool
            True if the node is terminal.
        """
        return node in self.payoffs

    def is_decision(self, node):
        """Check if node is a decision node (has children, not terminal).

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        bool
            True if the node is a decision node.
        """
        return (node in self.player and
                node not in self.payoffs and
                len(self.children.get(node, [])) > 0)

    def get_children(self, node):
        """Get children of a node.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        list of int
            List of child node IDs.
        """
        return self.children.get(node, [])


def backward_induction(game, node=0):
    """Solve a perfect-information extensive-form game via backward induction.

    Parameters
    ----------
    game : ExtensiveFormGame
        The game to solve.
    node : int
        Current node (defaults to root).

    Returns
    -------
    tuple
        (payoffs, strategy) where payoffs is a list of equilibrium payoffs
        and strategy is a dict mapping decision_node -> best_child.
    """
    if game.is_terminal(node):
        return list(game.payoffs[node]), {}

    current_player = game.player.get(node)
    # If node has no player assigned and is not terminal, walk through
    if current_player is None:
        child = game.get_children(node)[0]
        return backward_induction(game, child)

    children = game.get_children(node)

    if not children:
        return [0] * game.n_players, {}

    # Evaluate each child subtree once, storing results
    best_value = -np.inf
    best_child = None
    best_result = None

    for child in children:
        sub_payoffs, sub_strategy = backward_induction(game, child)
        payoff = sub_payoffs[current_player]

        if payoff > best_value:
            best_value = payoff
            best_child = child
            best_result = (sub_payoffs, sub_strategy)

    best_payoffs, best_strategy = best_result
    strategy = {node: best_child}
    strategy.update(best_strategy)
    return best_payoffs, strategy


def subgame_perfect_equilibrium(game, node=0):
    """Compute subgame perfect equilibrium for a perfect-information game.

    Equivalent to backward induction for perfect-information games.

    Parameters
    ----------
    game : ExtensiveFormGame
        The game to solve.
    node : int
        Current node (defaults to root).

    Returns
    -------
    tuple
        (payoffs, strategy_profile) where strategy_profile is a dict
        mapping each decision node to its optimal action (child node).
    """
    return backward_induction(game, node)


def compute_SPE(game):
    """Compute Subgame Perfect Equilibrium for sequential games.

    Convenience wrapper around subgame_perfect_equilibrium.

    Parameters
    ----------
    game : ExtensiveFormGame
        The extensive-form game.

    Returns
    -------
    tuple
        (payoffs, strategy) - equilibrium payoffs and strategy profile.
    """
    return subgame_perfect_equilibrium(game)


def entry_deterrence_game():
    """Create the classic Entry Deterrence game.

    Setup:
    1. Entrant decides: Enter or Stay Out.
    2. If Enter, Incumbent decides: Fight or Accommodate.
    Payoffs: (Enter, Accommodate) = (2, 2), (Enter, Fight) = (-1, -1),
    (Stay Out, _) = (0, 5)

    Returns
    -------
    ExtensiveFormGame
        The entry deterrence game.
    """
    game = ExtensiveFormGame(n_players=2)

    # Root: Entrant (player 0) chooses Enter or Stay Out
    enter_node, stay_out_node = game.add_decision_node(
        0, 0, 2, ["Enter", "Stay Out"]
    )

    # If Stay Out: payoffs (0, 5)
    game.add_terminal_node(stay_out_node, [0, 5], "result")

    # If Enter: Incumbent (player 1) chooses Fight or Accommodate
    fight_node, accom_node = game.add_decision_node(
        enter_node, 1, 2, ["Fight", "Accommodate"]
    )

    game.add_terminal_node(fight_node, [-1, -1], "result")
    game.add_terminal_node(accom_node, [2, 2], "result")

    return game


def centipede_game(n_stages=4, payoff_increment=1):
    """Create a Centipede game with alternating moves.

    Each player can either Take (end the game) or Pass (continue).
    Payoffs increase with each Pass.

    Parameters
    ----------
    n_stages : int
        Number of decision stages.
    payoff_increment : float
        Payoff increment per stage.

    Returns
    -------
    ExtensiveFormGame
        The centipede game.
    """
    game = ExtensiveFormGame(n_players=2)
    cur = 0

    for stage in range(n_stages):
        player = stage % 2
        t = stage

        # Classic centipede payoff structure:
        # At each stage, the player whose turn it is gets more by Taking
        # than they'd get if they Pass and the other player Takes next.
        if player == 0:
            take_pay = [t + 2, t]
        else:
            take_pay = [t, t + 2]

        if stage == n_stages - 1:
            children = game.add_decision_node(cur, player, 2, ["Take", "Pass"])
            game.add_terminal_node(children[0], take_pay, "Take")
            # Pass at last stage: give more to the OTHER player
            # so current player prefers to Take
            if player == 0:
                final_pay = [t + 1, t + 3]
            else:
                final_pay = [t + 3, t + 1]
            game.add_terminal_node(children[1], final_pay, "Pass")
        else:
            children = game.add_decision_node(cur, player, 2, ["Take", "Pass"])
            game.add_terminal_node(children[0], take_pay, "Take")
            cur = children[1]

    return game


def print_strategy(game, strategy):
    """Print the SPE strategy in a readable format.

    Parameters
    ----------
    game : ExtensiveFormGame
        The game.
    strategy : dict
        Strategy mapping node->child.
    """
    print("Subgame Perfect Equilibrium Strategy:")
    for node, child in sorted(strategy.items()):
        if game.is_decision(node):
            action = game.actions.get((node, child), "?")
            player = game.player[node]
            print(f"  At node {node} (Player {player+1}): choose '{action}'"
                  f" -> node {child}")
