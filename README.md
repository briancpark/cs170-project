# CS 170 Project Fall 2020

Take a look at the project spec before you get started!

Requirements:

Python 3.6+

You'll only need to install networkx to work with the starter code. For installation instructions, follow: https://networkx.github.io/documentation/stable/install.html

If using pip to download, run `python3 -m pip install networkx`


Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

When writing inputs/outputs:
- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
  - These are the functions run by the autograder to validate submissions


https://docs.google.com/document/d/1t239a30Y7fyx972lfCSCRv4xC_keaA5osfceB22YKQw/edit?ts=5fb1710d





# Reflection
Brian Park, Tony Kam, Jonah Noh, Alfonso Sanchez 

Your reflection should address the following questions:
* Describe (briefly and informally) the approach you used to generate your inputs, and what other approaches (if
any) you considered or would like to consider if given more time.
* Describe the algorithm you used for your solver. Why do you think it is a good approach?
* What other approaches did you try? How did they perform?
* What computational resources did you use? (e.g. AWS, instructional machines, etc.)
* If given more time, what else would you try to improve your approach?
Your reflection should be no more than 1000 words long, although it may be shorter as long as you respond to the questions above.

We considered using greedy algorithm such as the multiple knapsack algorithm to group students according to stress budget and maximizing happiness. In doing so, we started with k = n rooms, and then sought to cut down on the number of rooms by running the algorithm multiple times, with each consecutive iteration having less rooms than the previous one. However, we ran into an issue in terms of how the stress budget was calculated. Since the stress budget depends on the number of breakout rooms, it was difficult to try and compute the stress budget as the number of breakout rooms changed. Furthermore, we looked into packages and methods such as simulated annealing, but thought that the triple clique algorithm was the most feasible. We also considered working with MSTs and ratios, where each edge weight would be a ratio of the happiness and stress between two vertices (i.e. people).

We also tried a brute force algorithm that would enumerate all the cliques and sort through all the possible combinations for breakout rooms. Although it was computationally expensive, it was feasible enough for the smaller inputs. This placed us at the top of the leaderboard for the small-sized inputs, since the brute force algorithm was guaranteed to find the optimal solution given the alloted resources.

Computational resources we used were through our own machines. We worked with our own machines to debug our program easier. We could've saved execution time if we chose do use something like GCP or AWS, as these solvers did take a lot of computer power.

If given more time, we would try to improve our solver by searching more black box solvers in Python to make our job easier. The harder part would be in reducing our problem to be run under a premade algorithm or solver.

