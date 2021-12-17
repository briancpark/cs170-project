# CS 170 Project Fall 2020
Brian Park, Tony Kam, Jonah Noh, Alfonso Sanchez
## Getting Started

Create a new conda environment:
```sh
conda create -n cs170 python=3.9
conda activate cs170
```

Install requirements:
```sh
pip3 install -r requirements.txt
```

To run the solver:
```sh
python3 solver.py
```

To generate submission:
```sh
python3 prepare_submission.py outputs submission.json
```

## Brainstorming and Planning
[Google Doc](https://docs.google.com/document/d/1t239a30Y7fyx972lfCSCRv4xC_keaA5osfceB22YKQw/edit?ts=5fb1710d)
# Reflection
We considered using greedy algorithm such as the multiple knapsack algorithm to group students according to stress budget and maximizing happiness. In doing so, we started with k = n rooms, and then sought to cut down on the number of rooms by running the algorithm multiple times, with each consecutive iteration having less rooms than the previous one. However, we ran into an issue in terms of how the stress budget was calculated. Since the stress budget depends on the number of breakout rooms, it was difficult to try and compute the stress budget as the number of breakout rooms changed. Furthermore, we looked into packages and methods such as simulated annealing, but thought that the triple clique algorithm was the most feasible. We also considered working with MSTs and ratios, where each edge weight would be a ratio of the happiness and stress between two vertices (i.e. people).

We also tried a brute force algorithm that would enumerate all the cliques and sort through all the possible combinations for breakout rooms. Although it was computationally expensive, it was feasible enough for the smaller inputs. This placed us at the top of the leaderboard for the small-sized inputs, since the brute force algorithm was guaranteed to find the optimal solution given the alloted resources.

Computational resources we used were through our own machines. We worked with our own machines to debug our program easier. We could've saved execution time if we chose do use something like GCP or AWS, as these solvers did take a lot of computer power.

If given more time, we would try to improve our solver by searching more black box solvers in Python to make our job easier. The harder part would be in reducing our problem to be run under a premade algorithm or solver.

