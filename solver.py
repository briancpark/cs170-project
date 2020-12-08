import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_happiness
import sys
import utils
import parse
import itertools
from os.path import basename, normpath
import glob
import random


"""
    Documentation: hhttps://networkx.org/documentation/stable/index.html

"""


"""
    All that being said, to obtain an approximate solution, a greedy approach is, probably, the best heuristic: sort the items in decreasing order of size and insert them one by one into the first bin that has room for it. This heuristic is called first-fit decreasing. The main appeal of this heuristic is that we pack the big items first and hope that the little ones fill up the spaces.
"""


def greedyVersion(G, s):
    # Packing based on stress (fuck happiness, we don't care about that in cal)
    # As long as we find a working solution, it should result in a higher happiness than putting everyone into their own breakout room

    # Greedy Bin Packing Problem
    # Sort all edges based on stress

    # Each bin: We add
    """
    Sort all edges nlogn n(n-1)/2 edges complete graph
    pick a random student

    max_heap = store_all_stress edges.
    no_breakout = 1
    stressthreahold = stress/no_breakouts
    new_room_stressthreshold = stress/(no_breakout+1)

    breakout room pointer -> the current breakout room
    while (there are still students left):
        for (each student left):
            if we can add the student without pushing over the threshhold:
                we add the student
                take student out of added
    """
    return None


def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest


def greedy2(G, s):
    """
        check to see which two breakouts we can combine without exceeding the stress threshhold
        1. put everyone into their own breakouts
        2. go through all the breakouts and see if we can merge two together
        3. we do this until we can no longer 

    """
    d = {}
    i = 0
    for node in G.nodes:  # Place everyone in their own breakout room.
        d[i] = []
        d[i].append(node)
        i += 1
        # d[room #] : [list contains the students in the breakout room]
        # d[0] : [1, 2]
        # d[1] : [3, 4, 7]
        # d[2] : []
        # Mark item in dictionary to merge
    breakout_rooms = len(G.nodes)

    while(True):
        merge1 = None
        merge2 = None
        is_merge = False
        # First room
        ################
        # Generate a random number in 1-length
        d_temp1 = d.key()
        d_temp2 = d.key()

        d_temp1 = scrambled(d_temp1)
        d_temp2 = scrambled(d_temp2)

        for br1 in d_temp1:
            # Second room
            for br2 in d_temp2:
                # Two rooms have to be different
                if (d[br1] != d[br2]):
                    # Check to see if merge is possible.
                    temp = d[br1] + d[br2]
                    if (breakout_rooms != 1 and utils.calculate_stress_for_room(temp, G) < s / (breakout_rooms - 1)):
                        breakout_rooms -= 1
                        # mark the two items to merge
                        merge1 = br1
                        merge2 = br2
                        is_merge = True
                        break
            if (is_merge == True):
                break
        ################
        # modify breakout_room to prepare for next iteration
        # NO MORE POSSIBLE MERGES
        if is_merge == False:
            break
        else:
            #merge_rooms(deletion, merge1, merge2)
            if (merge1 == min(merge1, merge2)):
                d[merge1] = d[merge1]+d[merge2]
                del d[merge2]
            else:
                d[merge2] = d[merge1]+d[merge2]
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

    else:  # For medium/large inputs
        # return tripleClique(G, s)
        d1, b1 = tripleClique(G, s)
        d2, b2 = greedy2(G, s)
        d3, b3 = greedy2(G, s)
        if utils.calculate_happiness(d2, G) > utils.calculate_happiness(d3, G):
            if utils.calculate_happiness(d2, G) > utils.calculate_happiness(d1, G):
                return d2, b2
            else:
                return d1, b1
        else:
            if utils.calculate_happiness(d3, G) > utils.calculate_happiness(d1, G):
                return d3, b3
            else:
                return d1, b1


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

    partioned_sets_data.sort(key=lambda x: x[2], reverse=True)
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


def tripleClique(G, s):
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
    # This becomes the triple clique converted list
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
    # TODO:

    if len(duo_cliques) > 2:
        # print(len(duo_cliques))
        # if there more than 2 left over, asign them to each room.
        for duo_clique in duo_cliques:
            combined_set_list.append([duo_clique])
    else:
        # Fill it up with duoclique of 2 nodes
        combined_set_list.append(duo_cliques)

    # TODO:

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
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first` in its own subset
        yield [[first]] + smaller


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
# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
if __name__ == '__main__':
    #inputs = glob.glob('inputs/*')
    inputs = glob.glob('inputs/large/*.in')
    total = 0
    for input_path in inputs:
        output_path = 'outputs/large/' + \
            basename(normpath(input_path))[:-3] + '.out'
        G, s = read_input_file(input_path, 100)
        D, k = solve(G, s)
        try:
            assert is_valid_solution(D, G, s, k)
            write_output_file(D, output_path)
            print("success: " + str(input_path))
            total += 1
        except AssertionError as error:
            # brute force and assign everyone to their own breakout room for validity
            G, s = read_input_file(input_path, 100)
            D, k = greedy2(G, s)
            write_output_file(D, output_path)
            print("invalid - changed to Greedy: " + str(input_path))

        #cost_t = calculate_happiness(T)
    #print(total / len(inputs))
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G, s = read_input_file(input_path)
#         D, k = solve(G, s)
#         assert is_valid_solution(D, G, s, k)
#         happiness = calculate_happiness(D, G)
#         write_output_file(D, output_path)
