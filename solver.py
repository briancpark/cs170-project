"""
CS 170 Project Fall 2020

Usage:
    python solver.py                           # original solver with joblib
    python solver.py ray                       # original solver with Ray
    python solver.py --ortools                 # OR-Tools CP-SAT solver (all inputs)
    python solver.py --ortools inputs/small-1.in  # OR-Tools on a single input
"""

import argparse
import itertools
import os
import random
import glob
from os.path import basename, normpath
import networkx as nx
from tqdm import tqdm
import utils
from parse import read_input_file, write_output_file
from utils import is_valid_solution

# pylint: disable=invalid-name,too-many-branches,too-many-locals

random.seed(42)


# =============================================================================
# Original solver
# =============================================================================


def scrambled(orig):
    """Scramble a list"""
    random.seed(42)  # needs to be reseeded under parallelization
    dest = orig[:]
    random.shuffle(dest)
    return dest


def greedy2(G, s):
    """
    Check to see which two breakouts we can combine without exceeding the stress threshhold
    1. Put everyone into their own breakouts
    2. Go through all the breakouts and see if we can merge two together
    3. We do this until we can no longer merge
    """
    d = {}
    i = 0
    # Place everyone in their own breakout room.
    for node in G.nodes:
        d[i] = []
        d[i].append(node)
        i += 1
    breakout_rooms = len(G.nodes)

    while True:
        merge1 = None
        merge2 = None
        is_merge = False
        # Generate a random number in 1-length
        d_temp1 = list(d.keys())
        d_temp2 = list(d.keys())

        d_temp1 = scrambled(d_temp1)
        d_temp2 = scrambled(d_temp2)

        for br1 in d_temp1:
            # Second room
            for br2 in d_temp2:
                # Two rooms have to be different
                if d[br1] != d[br2]:
                    # Check to see if merge is possible.
                    temp = d[br1] + d[br2]
                    if breakout_rooms != 1 and utils.calculate_stress_for_room(
                        temp, G
                    ) < s / (breakout_rooms - 1):
                        breakout_rooms -= 1
                        # mark the two items to merge
                        merge1 = br1
                        merge2 = br2
                        is_merge = True
                        break
            if is_merge:
                break
        if not is_merge:
            break
        if merge1 == min(merge1, merge2):
            d[merge1] = d[merge1] + d[merge2]
            del d[merge2]
        else:
            d[merge2] = d[merge1] + d[merge2]
            del d[merge1]

    d_student_rooms = {}

    room_i = 0

    for room in d.values():
        for i in room:
            d_student_rooms[i] = room_i
        room_i += 1

    return d_student_rooms, breakout_rooms


def solve(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    """
    # If inputs are small, we run our naive method.
    if len(G.nodes) == 10:
        return naive(G, s)

    # For medium/large inputs
    d1, b1 = tripleClique(G, s)
    d2, b2 = greedy2(G, s)
    d3, b3 = greedy2(G, s)
    if utils.calculate_happiness(d2, G) > utils.calculate_happiness(d3, G):
        if utils.calculate_happiness(d2, G) > utils.calculate_happiness(d1, G):
            return d2, b2
        return d1, b1
    if utils.calculate_happiness(d3, G) > utils.calculate_happiness(d1, G):
        return d3, b3
    return d1, b1


def naive(G, s):
    """
    Brute Force method used for small inputs.
    Main idea: Try all combinations.
    Runtime: Very long.
    Note: This does not work for medium/large.
    """
    cliques = list(nx.enumerate_all_cliques(G))
    rooms_meta_data = {}
    for clique in cliques:
        stress = utils.calculate_stress_for_room(clique, G)
        happiness = utils.calculate_happiness_for_room(clique, G)
        rooms_meta_data[frozenset(clique)] = [stress, happiness]
    nodes = list(G.nodes)
    partioned_sets = []
    for _, p in enumerate(partition(nodes), 1):
        partioned_sets.append(sorted(p))
    partioned_sets_data = []
    for sets in partioned_sets:
        total_happiness = 0
        total_stress = 0
        for part in sets:
            total_happiness += rooms_meta_data[frozenset(part)][0]
            total_stress += rooms_meta_data[frozenset(part)][1]
        partioned_sets_data.append((sets, total_happiness, total_stress))
    deletion = []
    for partitioned_room in partioned_sets_data:
        for i in range(len(partitioned_room[0])):
            if s / len(partitioned_room[0]) < utils.calculate_stress_for_room(
                partitioned_room[0][i], G
            ):
                deletion.append(partitioned_room)
                continue
    for rem in deletion:
        if rem in partioned_sets_data:
            partioned_sets_data.remove(rem)

    partioned_sets_data.sort(key=lambda x: x[2], reverse=True)
    optimal = partioned_sets_data[0]

    d = {}

    for i in range(len(optimal[0])):
        for j in optimal[0][i]:
            d[j] = i

    return d, len(optimal[0])


def tripleClique(G, s):
    """
    In this function, we only create rooms of 3.
    That is a specific limitation that we've set to decrease computational complexity
    It prunes out certain combinations, but it's a decent approximation

    This is supposed to handle inputs of 20 and 50.
    To avoid the tailcase of dual clique, we add dummy people to turn our group size to
    a multiple of 3.

    If we're dealing with 20.in, we turn it into a group of 21.

    This triple case:

    Pre-process here

    double clique -> triple clique -> quad clique
    all clique are doubles, triples, or quad

    for one person's relationship with everyone
    * if all of them are bad
    * we put this person in their own room
    * and recrusively run the problem on the same group excluding this last,
      but decreasing the breakoutroom size by 1
    """

    if len(G.nodes) == 20:
        limit = s / 7
    else:
        limit = s / 17

    # Initializing all possible cliques
    triple_cliques = list(itertools.combinations(G, 3))

    # Initialize room data
    rooms_meta_data = {}

    # Calculate stress and happiness for each combination
    for clique in triple_cliques:
        stress = utils.calculate_stress_for_room(clique, G)
        happiness = utils.calculate_happiness_for_room(clique, G)
        rooms_meta_data[frozenset(clique)] = [stress, happiness]

    # Filter out everything that is not a valid clique
    # If the stress of a room is above the stress threshhold, then
    # that's not a valid solution.
    deletion = []
    for key, value in rooms_meta_data.items():
        if value[0] >= limit:
            deletion.append(key)
    for key in deletion:
        del rooms_meta_data[key]

    # Creating a list of all of the people
    nodes = list(G.nodes)

    # Our Answer: people assigned in BO rooms
    combined_set = []

    # Combinations we've seen before
    # This is used to figure out, what is the remainder for the final dual clique
    seen_sets = []
    i = 0
    # This becomes the triple clique converted list
    for vertices, _ in rooms_meta_data.items():
        breakTrue = False
        for vertex in list(vertices):
            if vertex in seen_sets:
                breakTrue = True
        if breakTrue:
            continue
        combined_set.append(vertices)
        for seen_v in list(vertices):
            seen_sets.append(seen_v)
    duo_cliques = [x for x in nodes if x not in seen_sets]
    combined_set_list = []
    for frozen_set in combined_set:
        combined_set_list.append(list(frozen_set))

    if len(duo_cliques) > 2:
        # If there more than 2 left over, assign them to each room.
        for duo_clique in duo_cliques:
            combined_set_list.append([duo_clique])
    else:
        # Fill it up with duoclique of 2 nodes
        combined_set_list.append(duo_cliques)

    # D is the dictionary of the breakout rooms in which each person is assigned to
    d = {}

    # Assign each clique into a room
    i = 0
    for clique in combined_set_list:
        for j in clique:
            d[j] = i
        i += 1

    return d, len(combined_set_list)


def partition(collection):
    """Partition a set into all possible subsets"""
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        # put `first` in its own subset
        yield [[first]] + smaller


def solve_wrapper(input_path):
    """Wrapper function to be parallelized"""
    output_path = "outputs/" + basename(normpath(input_path))[:-3] + ".out"
    G, s = read_input_file(input_path, 100)
    D, k = solve(G, s)
    try:
        assert is_valid_solution(D, G, s, k)
        write_output_file(D, output_path)
    except AssertionError:
        # brute force and assign everyone to their own breakout room for validity
        G, s = read_input_file(input_path, 100)
        D, k = greedy2(G, s)
        write_output_file(D, output_path)


# =============================================================================
# OR-Tools CP-SAT solver
# =============================================================================

SCALE = 1000  # Float -> int scale (input values have <= 3 decimal places)


def solve_for_k(G, s, k, time_limit=10, hint=None):
    """
    Solve the breakout room problem for a fixed number of rooms k using CP-SAT.

    Args:
        G: networkx.Graph with 'happiness' and 'stress' edge attributes
        s: stress budget (S_max)
        k: number of rooms
        time_limit: solver time limit in seconds
        hint: optional dict mapping student -> room (warm start)

    Returns:
        (D, happiness) or (None, 0) if infeasible
    """
    from ortools.sat.python import cp_model

    nodes = sorted(G.nodes)

    # Collect unique pairs (i < j)
    pairs = []
    for u, v in G.edges():
        i, j = (u, v) if u < v else (v, u)
        pairs.append((i, j))
    pairs = sorted(set(pairs))

    s_max_scaled = int(round(s * SCALE))

    model = cp_model.CpModel()

    # --- Decision variables: x[i, r] = 1 if student i in room r ---
    x = {}
    for i in nodes:
        for r in range(k):
            x[i, r] = model.new_bool_var(f"x_{i}_{r}")
        model.add_exactly_one(x[i, r] for r in range(k))

    # --- Symmetry breaking: first student always in room 0 ---
    model.add(x[nodes[0], 0] == 1)

    # --- Pair-room indicators: y[i,j,r] = 1 iff both i,j in room r ---
    y = {}
    for i, j in pairs:
        for r in range(k):
            y[i, j, r] = model.new_bool_var(f"y_{i}_{j}_{r}")
            model.add(y[i, j, r] <= x[i, r])
            model.add(y[i, j, r] <= x[j, r])
            model.add(y[i, j, r] >= x[i, r] + x[j, r] - 1)

    # --- Stress constraints per room ---
    # k * sum(s_ij * y_ijr) <= S_max  (avoids dividing S_max by k)
    for r in range(k):
        model.add(
            sum(
                k * int(round(G[i][j]["stress"] * SCALE)) * y[i, j, r] for i, j in pairs
            )
            <= s_max_scaled
        )

    # --- Objective: maximize total happiness ---
    model.maximize(
        sum(
            int(round(G[i][j]["happiness"] * SCALE)) * y[i, j, r]
            for i, j in pairs
            for r in range(k)
        )
    )

    # --- Warm start from greedy solution ---
    if hint is not None:
        for i in nodes:
            room = hint.get(i)
            if room is not None:
                for r in range(k):
                    model.add_hint(x[i, r], 1 if room == r else 0)

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = os.cpu_count() or 8
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        D = {}
        for i in nodes:
            for r in range(k):
                if solver.value(x[i, r]):
                    D[i] = r
                    break
        return D, solver.objective_value / SCALE

    return None, 0


def greedy_solve(G, s):
    """
    Quick greedy baseline: start with everyone alone, merge rooms greedily.
    Returns (D, k) where D maps student -> room.
    """
    from utils import calculate_stress_for_room

    nodes = sorted(G.nodes)
    rooms = {i: [node] for i, node in enumerate(nodes)}
    k = len(nodes)

    improved = True
    while improved:
        improved = False
        room_keys = list(rooms.keys())
        for idx1 in range(len(room_keys)):
            for idx2 in range(idx1 + 1, len(room_keys)):
                r1, r2 = room_keys[idx1], room_keys[idx2]
                if r1 not in rooms or r2 not in rooms:
                    continue
                merged = rooms[r1] + rooms[r2]
                new_k = k - 1
                if new_k == 0:
                    continue
                if calculate_stress_for_room(merged, G) <= s / new_k:
                    rooms[r1] = merged
                    del rooms[r2]
                    k = new_k
                    improved = True
                    break
            if improved:
                break

    D = {}
    for room_idx, (_, students) in enumerate(rooms.items()):
        for student in students:
            D[student] = room_idx
    return D, k


def remap_hint(hint, k):
    """Remap a hint dict so all room values are in [0, k)."""
    if hint is None:
        return None
    unique_rooms = sorted(set(hint.values()))
    if len(unique_rooms) != k:
        return None
    room_map = {old: new for new, old in enumerate(unique_rooms)}
    return {student: room_map[room] for student, room in hint.items()}


def solve_ortools(G, s):
    """
    OR-Tools solver: tries different k values with CP-SAT and returns the best.
    Uses greedy solution as warm start hint.
    """
    from utils import calculate_happiness

    n = len(G.nodes)

    # Quick greedy baseline
    greedy_D, greedy_k = greedy_solve(G, s)
    greedy_happiness = calculate_happiness(greedy_D, G)

    best_D = greedy_D
    best_k = greedy_k
    best_happiness = greedy_happiness

    # Configure search range and time limits by input size
    if n <= 10:
        k_values = list(range(1, n + 1))
        time_per_k = 2
    elif n <= 20:
        k_min = max(1, greedy_k - 3)
        k_max = min(n, greedy_k + 4)
        k_values = list(range(k_min, k_max + 1))
        time_per_k = 5
    else:  # n <= 50
        k_min = max(1, greedy_k - 2)
        k_max = min(n, greedy_k + 3)
        k_values = list(range(k_min, k_max + 1))
        time_per_k = 10

    for k in k_values:
        hint = remap_hint(greedy_D, k) if greedy_k == k else None
        D, happiness = solve_for_k(G, s, k, time_per_k, hint)

        if D is not None and happiness > best_happiness:
            num_rooms = len(set(D.values()))
            if is_valid_solution(D, G, s, num_rooms):
                best_D = D
                best_k = num_rooms
                best_happiness = happiness

    return best_D, best_k


def solve_ortools_wrapper(input_path):
    """Process a single input file with OR-Tools."""
    from utils import calculate_happiness

    output_path = "outputs/" + basename(normpath(input_path))[:-3] + ".out"
    G, s = read_input_file(input_path, 100)
    D, k = solve_ortools(G, s)
    assert is_valid_solution(D, G, s, k), f"Invalid solution for {input_path}"
    happiness = calculate_happiness(D, G)
    write_output_file(D, output_path)
    return input_path, happiness, k


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS 170 Breakout Room Solver")
    parser.add_argument(
        "--ortools", action="store_true", help="Use OR-Tools CP-SAT solver"
    )
    parser.add_argument(
        "--ray", action="store_true", help="Use Ray backend (original solver only)"
    )
    parser.add_argument(
        "input_file", nargs="?", default=None, help="Single input file to solve"
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if args.ortools:
        # ---- OR-Tools backend ----
        if args.input_file and os.path.isfile(args.input_file):
            print(f"Solving {args.input_file} with OR-Tools...")
            _, happiness, k = solve_ortools_wrapper(args.input_file)
            print(f"  Happiness: {happiness:.3f}, Rooms: {k}")
        else:
            inputs = sorted(glob.glob("inputs/*"))
            print(f"Processing {len(inputs)} inputs with OR-Tools CP-SAT...")
            total_happiness = 0
            for path in tqdm(inputs):
                try:
                    _, happiness, k = solve_ortools_wrapper(path)
                    total_happiness += happiness
                except Exception as e:
                    print(f"\nError on {path}: {e}")
            print(f"\nTotal happiness across all inputs: {total_happiness:.3f}")
    else:
        # ---- Original backend ----
        inputs = sorted(glob.glob("inputs/*"))

        if args.ray:
            import ray

            ray.init()
            solve_wrapper_remote = ray.remote(solve_wrapper)
            oids = [solve_wrapper_remote.remote(path) for path in inputs]
            for _ in tqdm(oids, total=len(oids)):
                ray.get(_)
            ray.shutdown()
        else:
            from joblib import Parallel, delayed

            Parallel(n_jobs=-1)(delayed(solve_wrapper)(path) for path in tqdm(inputs))
