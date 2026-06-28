"""
Matching Markets Module
=======================

Provides implementations of classic matching algorithms, including the
Gale-Shapley deferred acceptance algorithm, Top Trading Cycles (TTC),
school choice, and stability verification.

Algorithms implemented:
- Gale-Shapley deferred acceptance (men-proposing and women-proposing)
- School choice (student-proposing DA with capacities)
- Top Trading Cycles (TTC)
- Random Serial Dictatorship (RSD)
- Stability verification
- Random preference generation
"""

import numpy as np
from collections import deque


def deferred_acceptance(men_prefs, women_prefs, proposing_side='men'):
    """Gale-Shapley deferred acceptance algorithm for stable matching.

    Parameters
    ----------
    men_prefs : list of list
        Preference lists for men. men_prefs[i] is man i's ranking of women.
        Lower indices = higher preference.
    women_prefs : list of list
        Preference lists for women.
    proposing_side : str
        'men' for men-proposing, 'women' for women-proposing.

    Returns
    -------
    dict
        Matching from proposing side to receiving side.
        Keys are indices of the proposing side, values are indices
        of the receiving side (or None if unmatched).
    """
    if proposing_side == 'women':
        # Reverse roles
        matching = _da_core(women_prefs, men_prefs)
        # Flip keys/values
        return {v: k for k, v in matching.items() if v is not None}
    else:
        return _da_core(men_prefs, women_prefs)


def _da_core(proposers_prefs, receivers_prefs):
    """Core deferred acceptance implementation.

    Parameters
    ----------
    proposers_prefs : list of list
        Preference lists for proposers.
    receivers_prefs : list of list
        Preference lists for receivers.

    Returns
    -------
    dict
        Matching from proposer indices to receiver indices.
    """
    n_proposers = len(proposers_prefs)
    n_receivers = len(receivers_prefs)

    # Build ranking dictionaries for receivers
    receiver_rankings = []
    for prefs in receivers_prefs:
        ranking = {p: rank for rank, p in enumerate(prefs)}
        receiver_rankings.append(ranking)

    # Current matching for each receiver (None = unmatched)
    receiver_match = {r: None for r in range(n_receivers)}

    # Index of next proposal for each proposer
    next_proposal = [0] * n_proposers

    # Queue of unmatched proposers
    unmatched = deque(range(n_proposers))

    while unmatched:
        proposer = unmatched.popleft()

        if next_proposal[proposer] >= len(proposers_prefs[proposer]):
            # Ran out of options; proposer remains unmatched
            continue

        # Next preferred receiver
        receiver = proposers_prefs[proposer][next_proposal[proposer]]
        next_proposal[proposer] += 1

        if receiver_match[receiver] is None:
            # Receiver is free: match
            receiver_match[receiver] = proposer
        else:
            current_match = receiver_match[receiver]
            ranking = receiver_rankings[receiver]

            # Check if new proposer is preferred over current match
            if ranking.get(proposer, float('inf')) < ranking.get(current_match, float('inf')):
                # New proposer preferred: switch
                receiver_match[receiver] = proposer
                unmatched.append(current_match)
            else:
                # Current match preferred: proposer stays unmatched
                unmatched.append(proposer)

    # Convert to proposer->receiver mapping
    matching = {}
    for r, p in receiver_match.items():
        if p is not None:
            matching[p] = r

    # Proposers who didn't get matched
    for p in range(n_proposers):
        if p not in matching:
            matching[p] = None

    return matching


def school_choice(students, schools, capacities):
    """Student-proposing deferred acceptance for school choice.

    Each student has preferences over schools. Each school has a priority
    ordering over students and a capacity.

    Parameters
    ----------
    students : list of list
        Student preferences. students[i] = [ranked school indices].
    schools : list of list
        School priorities. schools[j] = [ranked student indices].
    capacities : list of int
        Capacity for each school.

    Returns
    -------
    dict
        Mapping from student index to school index (or None if unmatched).
    list of list
        List of students assigned to each school.
    """
    n_students = len(students)
    n_schools = len(schools)

    # Build school priority rankings
    school_ranking = []
    for priority in schools:
        ranking = {s: rank for rank, s in enumerate(priority)}
        school_ranking.append(ranking)

    # Assignments
    school_assignments = {s: [] for s in range(n_schools)}
    student_assignment = {s: None for s in range(n_students)}
    next_proposal = [0] * n_students

    unmatched = deque(range(n_students))

    while unmatched:
        student = unmatched.popleft()

        if (next_proposal[student] >= len(students[student]) or
                len(students[student]) == 0):
            continue

        school = students[student][next_proposal[student]]
        next_proposal[student] += 1

        if len(school_assignments[school]) < capacities[school]:
            # School has space
            school_assignments[school].append(student)
            student_assignment[student] = school
        else:
            # School is full: find lowest-priority admitted student
            current_students = school_assignments[school]
            ranking = school_ranking[school]

            # Find the student with worst priority
            worst_student = None
            worst_rank = -1
            for s in current_students:
                r = ranking.get(s, float('inf'))
                if r > worst_rank:
                    worst_rank = r
                    worst_student = s

            student_rank = ranking.get(student, float('inf'))

            if student_rank < worst_rank:
                # Admit new student, reject worst current
                school_assignments[school].remove(worst_student)
                school_assignments[school].append(student)
                student_assignment[student] = school
                student_assignment[worst_student] = None
                unmatched.append(worst_student)
            else:
                unmatched.append(student)

    return student_assignment, [school_assignments[s] for s in range(n_schools)]


def top_trading_cycles(students, schools, capacities, priorities):
    """Top Trading Cycles (TTC) algorithm for school choice.

    Each student points to their most preferred school with capacity.
    Each school points to its highest-priority student.
    Cycles are resolved by assignment and removal.

    Parameters
    ----------
    students : list of list
        Student preferences over schools.
    schools : list of list
        School priorities over students.
    capacities : list of int
        Capacity for each school.
    priorities : list of list
        Same as schools, priority ordering of students per school.

    Returns
    -------
    dict
        Mapping from student index to school index (or None).
    """
    n_students = len(students)
    n_schools = len(schools)

    remaining_capacity = list(capacities)
    remaining_students = set(range(n_students))
    remaining_schools = set(range(n_schools))

    # Make copies of student preferences, filtering out unavailable schools
    student_prefs = [list(prefs) for prefs in students]

    assignment = {s: None for s in range(n_students)}

    while remaining_students:
        # Each remaining student points to most preferred school with capacity
        pointing_to = {}
        for s in remaining_students:
            for school in student_prefs[s]:
                if school in remaining_schools and remaining_capacity[school] > 0:
                    pointing_to[s] = school
                    break

        # Each school points to highest-priority remaining student
        school_points_to = {}
        for school in remaining_schools:
            if remaining_capacity[school] <= 0:
                continue
            for s in priorities[school]:
                if s in remaining_students:
                    school_points_to[school] = s
                    break

        # Find a single cycle
        cycle = _find_ttc_cycle(pointing_to, school_points_to, remaining_students)

        if cycle is None:
            break  # No more cycles

        # Resolve the cycle
        for student in cycle:
            school = pointing_to[student]
            assignment[student] = school
            remaining_capacity[school] -= 1
            if remaining_capacity[school] == 0:
                remaining_schools.discard(school)
        for student in cycle:
            remaining_students.discard(student)

    return assignment


def _find_ttc_cycle(pointing_to, school_points_to, valid_students):
    """Find one cycle in the TTC pointing graph.

    Parameters
    ----------
    pointing_to : dict
        Mapping from student to school.
    school_points_to : dict
        Mapping from school to student.
    valid_students : set
        Set of students still in the game.

    Returns
    -------
    list or None
        List of students forming a cycle, or None if no cycle exists.
    """
    for start in valid_students:
        if start not in pointing_to:
            continue
        current = start
        seen = {start: 0}
        path = [start]

        while True:
            school = pointing_to.get(current)
            if school is None or school not in school_points_to:
                break
            next_student = school_points_to[school]
            if next_student is None:
                break
            if next_student in seen:
                # Found a cycle
                return path[seen[next_student]:]
            seen[next_student] = len(path)
            path.append(next_student)
            current = next_student

    return None


def is_stable(matching, prefs, side='men'):
    """Check whether a matching is stable.

    A matching is stable if there is no blocking pair: a man m and woman w
    who are not matched to each other, but prefer each other to their
    current matches.

    Parameters
    ----------
    matching : dict
        Mapping from men to women (or proposers to receivers).
    prefs : tuple of list
        (side1_prefs, side2_prefs) as lists of preference lists.
    side : str
        'men' or 'women' indicating the key side.

    Returns
    -------
    tuple
        (is_stable: bool, blocking_pairs: list)
    """
    men_prefs, women_prefs = prefs

    if side == 'men':
        m_to_w = matching
        w_to_m = {v: k for k, v in matching.items() if v is not None}

        # Build rankings
        men_ranking_w = []
        for prefs_list in men_prefs:
            men_ranking_w.append({w: rank for rank, w in enumerate(prefs_list)})
        women_ranking_m = []
        for prefs_list in women_prefs:
            women_ranking_m.append({m: rank for rank, m in enumerate(prefs_list)})

        blocking_pairs = []
        for m, w_matched in m_to_w.items():
            if w_matched is None:
                continue
            m_rank_current = men_ranking_w[m].get(w_matched, float('inf'))
            # Check women that m prefers over current match
            for w in men_prefs[m]:
                if men_ranking_w[m][w] >= m_rank_current:
                    break  # Since prefs are ranked in order
                # Check if w prefers m to her current match
                w_current = w_to_m.get(w)
                if w_current is None:
                    blocking_pairs.append((m, w))
                else:
                    w_rank_m = women_ranking_m[w].get(m, float('inf'))
                    w_rank_current = women_ranking_m[w].get(w_current, float('inf'))
                    if w_rank_m < w_rank_current:
                        blocking_pairs.append((m, w))

        return len(blocking_pairs) == 0, blocking_pairs

    return True, []


def random_serial_dictatorship(agents, items, preferences=None):
    """Random Serial Dictatorship for fair allocation.

    Agents are randomly ordered. Each agent picks their most preferred
    remaining item in turn.

    Parameters
    ----------
    agents : int
        Number of agents.
    items : int
        Number of items.
    preferences : list of list, optional
        Agent preferences over items. If None, random preferences generated.

    Returns
    -------
    dict
        Mapping from agent index to item index (or None if no items left).
    """
    if preferences is None:
        preferences = generate_random_preferences(agents, items)

    ordering = list(range(agents))
    np.random.shuffle(ordering)

    available = set(range(items))
    allocation = {a: None for a in range(agents)}

    for agent in ordering:
        for item in preferences[agent]:
            if item in available:
                allocation[agent] = item
                available.discard(item)
                break

    return allocation


def generate_random_preferences(n_agents, n_items, seed=None):
    """Generate random preference lists for matching simulations.

    Parameters
    ----------
    n_agents : int
        Number of agents.
    n_items : int
        Number of items.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of list
        Random preference lists for each agent.
    """
    rng = np.random.RandomState(seed)
    preferences = []
    for _ in range(n_agents):
        prefs = list(range(n_items))
        rng.shuffle(prefs)
        preferences.append(prefs)
    return preferences
