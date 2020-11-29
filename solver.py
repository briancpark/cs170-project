import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_happiness
import sys
<<<<<<< HEAD
import utils
import parse
import itertools
from os.path import basename, normpath
import glob


def solve(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    """

    if len(G.nodes) == 10:
        cliques = list(nx.enumerate_all_cliques(G))

        rooms_meta_data = {}

        for clique in cliques:
            stress = utils.calculate_stress_for_room(clique, G)
            happiness = utils.calculate_happiness_for_room(clique, G)
            rooms_meta_data[frozenset(clique)] = [stress, happiness]

        nodes = [node for node in G.nodes]

        partioned_sets = []

        for n, p in enumerate(partition(nodes), 1):
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
                if s / len(partitioned_room[0]) < utils.calculate_stress_for_room(partitioned_room[0][i], G):
                    deletion.append(partitioned_room)
                    continue
                
        for rem in deletion:
            if rem in partioned_sets_data:
                partioned_sets_data.remove(rem)

        partioned_sets_data.sort(key=lambda x:x[2], reverse=True)
        optimal = partioned_sets_data[0]

        d = {}

        for i in range(len(optimal[0])):
            for j in optimal[0][i]:
                d[j] = i

        return d, len(optimal[0])
    else:

        if len(G.nodes) == 20:
            limit = s / 7
        else:
            limit = s / 17

        triple_cliques = list(itertools.combinations(G, 3))

        rooms_meta_data = {}

        for clique in triple_cliques:
            stress = utils.calculate_stress_for_room(clique, G)
            happiness = utils.calculate_happiness_for_room(clique, G)
            rooms_meta_data[frozenset(clique)] = [stress, happiness]

        deletion = []

        for key in rooms_meta_data:
            if rooms_meta_data[key][0] > limit:
                deletion.append(key)
            
        for key in deletion:
            del rooms_meta_data[key]

        nodes = [node for node in G.nodes]


        combined_set = [] # this contains our answer
        seen_sets = [] #add seen here
        i = 0

        #This becomes the triple clique converted list
        for vertices in rooms_meta_data.keys():
            breakTrue = False
            for vertex in list(vertices):
                if vertex in seen_sets:
                    breakTrue = True

            if breakTrue:
                continue

            combined_set.append(vertices)
            for seen_v in list(vertices):
                seen_sets.append(seen_v)

        duo_clique = [x for x in nodes if x not in seen_sets]

        duo_clique
        utils.calculate_stress_for_room(duo_clique, G)

        combined_set_list = []


        for frozen_set in combined_set:
            combined_set_list.append(list(frozen_set))

        combined_set_list.append(duo_clique)
        combined_set_list

        d = {}
        i = 0

        for clique in combined_set_list:
            for j in clique:
                d[j] = i
            i += 1

        return d, len(combined_set_list)


def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller



# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G, s = read_input_file(path)
#     D, k = solve(G, s)
#     assert is_valid_solution(D, G, s, k)
#     print("Total Happiness: {}".format(calculate_happiness(D, G)))
#     write_output_file(D, 'outputs/small-1.out')

import glob
from os.path import basename, normpath
# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
if __name__ == '__main__':
    #inputs = glob.glob('inputs/*')
    inputs = glob.glob('inputs/small/*.in')
    for input_path in inputs:
        output_path = 'outputs/small/' + basename(normpath(input_path))[:-3] + '.out'
        G, s = read_input_file(input_path, 100)
        D, k = solve(G, s)
        assert is_valid_solution(D, G, s, k)
        #cost_t = calculate_happiness(T)
        write_output_file(D, output_path)

# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G, s = read_input_file(input_path)
#         D, k = solve(G, s)
#         assert is_valid_solution(D, G, s, k)
#         happiness = calculate_happiness(D, G)
#         write_output_file(D, output_path)
