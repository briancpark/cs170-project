import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_happiness
import sys
import utils
import parse
import itertools
from os.path import basename, normpath
import glob


"""
    Documentation: hhttps://networkx.org/documentation/stable/index.html

"""


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
        return naive(G,s)
    
    else: #For medium/large inputs
        return tripleClique(G,s)
    
"""
    Brute Force method used for small inputs.
    Main idea: Try all combinations.
    Runtime: Very long.
    Note: This does not work for medium/large.
"""
def naive(G, s):
    # If the size is 10, then we use brute force method.
    
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


"""
    Triple Clique method used for medium and large inputs.
    Main idea: 
    Runtime:
"""
def tripleClique(G,s):
    # In this function, we only create rooms of 3.
    # That is a specific limitation that we've set to decrease computational complexity
    # It prunes out certain combinations, but it's a decent approximation
    
    # This is supposed to handle inputs of 20 and 50.
    # To avoid the tailcase of dual clique, we add dummy people to turn our group size to
    # a multiple of 3.

    # If we're dealing with 20.in, we turn it into a group of 21.
    
    # This triple case: 

    # Pre-process here

    # double clique -> triple clique -> quad clique
    # all clique are doubles, triples, or quad

    # for one person's relationship with everyone
    #   if all of them are bad
    #   we put this person in their own room
    #   and recrusively run the problem on the same group excluding this last, but decreasing the breakoutroom size by 1
    
    if len(G.nodes) == 20:
        limit = s / 7
    else:
        limit = s / 17

    # Initializing all possible cliques
    triple_cliques = list(itertools.combinations(G, 3))

    # initialize room data
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
    for key in rooms_meta_data:
        if rooms_meta_data[key][0] >= limit:
            deletion.append(key)
    for key in deletion:
        del rooms_meta_data[key]
    
    # Creating a list of all of the people
    nodes = [node for node in G.nodes]
    
    # Our Answer: people assigned in BO rooms
    combined_set = []
    
    # Combinations we've seen before
    # This is used to figure out, what is the remainder for the final dual clique
    seen_sets = [] 
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
    duo_cliques = [x for x in nodes if x not in seen_sets]
    duo_cliques
    #utils.calculate_stress_for_room(duo_cliques, G)
    combined_set_list = []
    for frozen_set in combined_set:
        combined_set_list.append(list(frozen_set))
    ### TODO:


    if len(duo_cliques) > 2:
        #print(len(duo_cliques))
        #if there more than 2 left over, asign them to each room.
        for duo_clique in duo_cliques:
            combined_set_list.append([duo_clique])
    else:
        #Fill it up with duoclique of 2 nodes
        combined_set_list.append(duo_cliques)


    ### TODO:

    
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
    inputs = glob.glob('inputs/large/*.in')
    total = 0
    for input_path in inputs:
        output_path = 'outputs/large/' + basename(normpath(input_path))[:-3] + '.out'
        G, s = read_input_file(input_path, 100)
        D, k = solve(G, s)
        try:
            assert is_valid_solution(D, G, s, k)
            write_output_file(D, output_path)
            print("success: " + str(input_path))
            total += 1
        except AssertionError as error:
            #brute force and assign everyone to their own breakout room for validity
            G, s = read_input_file(input_path, 100)
            D = {}
            i = 0 #the breakout room
            for node in G.nodes:
                D[node] = i
                i += 1
            write_output_file(D, output_path)
            print("invalid: " + str(input_path))

        #cost_t = calculate_happiness(T)
    print(total / len(inputs))
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G, s = read_input_file(input_path)
#         D, k = solve(G, s)
#         assert is_valid_solution(D, G, s, k)
#         happiness = calculate_happiness(D, G)
#         write_output_file(D, output_path)
