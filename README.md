# CS 170 Project Fall 2020

Brian Park, Tony Kam, Jonah Noh, Alfonso Sanchez

**Final ranking: 9th out of 243 teams**

## Getting Started

```sh
conda create -n cs170 python=3.9
conda activate cs170
pip install -r requirements.txt
```

## Usage

Run the solver:

```sh
python solver.py                              # Original solver (Joblib)
python solver.py --ray                        # Original solver (Ray)
python solver.py --ortools                    # OR-Tools CP-SAT solver (all inputs)
python solver.py --ortools inputs/small-1.in  # OR-Tools on a single input
python solver.py --ortools --size medium large # OR-Tools on specific sizes
```

Visualize outputs:

```sh
python visualize.py                           # Summary + one example per size -> viz/
python visualize.py outputs/small-1.out       # Single file (interactive)
python visualize.py --size small --save viz   # All small outputs -> viz/
python visualize.py --summary                 # Summary dashboard only
```

Prepare submission:

```sh
python prepare_submission.py outputs submission.json
```
