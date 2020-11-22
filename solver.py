import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_happiness
import sys
import utils
import parse



def solve(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    """
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



def brute_force_solve(G, s):
    """
    Main idea: We go through all possible numbers of breakout rooms (1→ n). For each number of breakout rooms, we try every possible permutation of students. We calculate the amount of stress for each one of these methods. If it is below the threshold limit set based on the number of breakout rooms, then we count it as a valid solution. We then compare all valid solutions and pick the solution with the largest happiness value.

    """
    num_students = G.number_of_nodes()
    # happiness & stress is a dictionary where key is tuple and the corresponding value is the stress
    # for example {(0, 1): 9.2, (0, 2): 5.4, (0, 3): 2.123, (1, 2): 75.4, (1, 3): 18.0, (2, 3): 87.0}
    # the tuple respresents the edge.
    happiness = nx.get_edge_attributes(G,'happiness')
    stress = nx.get_edge_attributes(G,'stress')
    
    for breakout_room in range(1, num_students): #Breakout rooms 1 -> n
        for group in range(breakout_room): #Go through each breakout room
            for edge in G.edges: # 
                
                # Total accumulated stress should not exceed s / k here
                if 0 < s:

                    happiness[edge] = 0
                    stress[edge] =0



    return None #Supposed to reutrn D, k



# For 4 people
    # One big breakout rooms.
        # [[1,2,3,4]] 

    # Two breakout rooms.
        # [[1,2][3,4]] 
        # [[1,3][2,4]]
        # [[1,4],[2,3]]
        ##############
        # [[1][2,3,4]]  [[2,3,4][1]]
        # [[2][1,3,4]]  [[1,3,4][2]]
        # [[3][1,2,4]]  [[1,2,4][3]]
        # [[4][1,2,3]]  [[1,2,3][4]]
        ##############

    # Three breakout rooms.
        # [[1][2][3,4]]
        # [[1][3][2,4]]
        # [[1][4][2,3]]
        # [[2][3][1,4]]
        # [[2][3][1,4]]
        

    # Four breakout rooms.
        # [[1][2][3][4]]

# From https://github.com/networkx/networkx/blob/master/networkx/algorithms/clique.py
def enumerate_all_cliques(G):
    """Returns all cliques in an undirected graph.
    This function returns an iterator over cliques, each of which is a
    list of nodes. The iteration is ordered by cardinality of the
    cliques: first all cliques of size one, then all cliques of size
    two, etc.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    iterator
        An iterator over cliques, each of which is a list of nodes in
        `G`. The cliques are ordered according to size.
    Notes
    -----
    To obtain a list of all cliques, use
    `list(enumerate_all_cliques(G))`. However, be aware that in the
    worst-case, the length of this list can be exponential in the number
    of nodes in the graph (for example, when the graph is the complete
    graph). This function avoids storing all cliques in memory by only
    keeping current candidate node lists in memory during its search.
    The implementation is adapted from the algorithm by Zhang, et
    al. (2005) [1]_ to output all cliques discovered.
    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.
    References
    ----------
    .. [1] Yun Zhang, Abu-Khzam, F.N., Baldwin, N.E., Chesler, E.J.,
           Langston, M.A., Samatova, N.F.,
           "Genome-Scale Computational Approaches to Memory-Intensive
           Applications in Systems Biology".
           *Supercomputing*, 2005. Proceedings of the ACM/IEEE SC 2005
           Conference, pp. 12, 12--18 Nov. 2005.
           <https://doi.org/10.1109/SC.2005.29>.
    """
    index = {}
    nbrs = {}
    for u in G:
        index[u] = len(index)
        # Neighbors of u that appear after u in the iteration order of G.
        nbrs[u] = {v for v in G[u] if v not in index}

    queue = deque(([u], sorted(nbrs[u], key=index.__getitem__)) for u in G)
    # Loop invariants:
    # 1. len(base) is nondecreasing.
    # 2. (base + cnbrs) is sorted with respect to the iteration order of G.
    # 3. cnbrs is a set of common neighbors of nodes in base.
    while queue:
        base, cnbrs = map(list, queue.popleft())
        yield base
        for i, u in enumerate(cnbrs):
            # Use generators to reduce memory consumption.
            queue.append(
                (
                    chain(base, [u]),
                    filter(nbrs[u].__contains__, islice(cnbrs, i + 1, None)),
                )
            )


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G, s = read_input_file(path)
    D, k = solve(G, s)
    assert is_valid_solution(D, G, s, k)
    print("Total Happiness: {}".format(calculate_happiness(D, G)))
    write_output_file(D, 'out/test.out')

# import glob
# from os.path import basename, normpath
# # For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('in_out/*')
#     for input_path in inputs:
#         output_path = 'out/' + basename(normpath(input_path))[:-3] + '.out'
#         G, s = read_input_file(input_path, 100)
#         D, k = solve(G, s)
#         assert is_valid_solution(D, G, s, k)
#         cost_t = calculate_happiness(T)
#         write_output_file(D, output_path)