# Route Optimization with a Genetic Algorithm 

A permutation-based Genetic Algorithm that finds a near-optimal closed tour
through *N* cities (visit each exactly once, return to the origin) while keeping
total travel **time** within `Tmax` and total **cost** within `Cmax`.
The bundled input now uses approximate kilometre coordinates for Indian cities.

> Course: M.Tech AIML - Artificial & Computational Intelligence (AIMLCZG557)

## Problem

Given city coordinates plus a time budget (`Tmax`) and a cost budget (`Cmax`),
search for the shortest / fastest round-trip that satisfies both budgets. With
20 cities the search space is far too large to brute-force, so the GA evolves a
population of candidate routes toward a good solution.

## Techniques implemented

| Requirement        | Where in the code        |
|--------------------|--------------------------|
| Permutation encoding | `init_population`      |
| Ordered crossover (OX) | `ordered_crossover`  |
| Swap mutation        | `swap_mutation`        |
| Rank selection       | `rank_selection`       |

Baselines for comparison: **random search** (`random_search`) and
**hill climbing** (`hill_climbing`).

## Repository layout

```
route_ga.py        # single, self-contained solution (the deliverable)
inputPS15.txt      # sample problem definition (Tmax, Cmax, city coordinates)
outputPS15.txt     # generated results report
best_route.png             # best-route map
distance_convergence.png   # best distance vs generation
time_convergence.png       # best time vs generation
README.md
```

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`

```bash
pip install numpy matplotlib
```

## Running

```bash
python route_ga.py inputPS15.txt outputPS15.txt
```

If the input file is omitted, the program generates a random 20-city instance
so it still runs. To grade against a different instance, just replace
`inputPS15.txt` - nothing is hardcoded.

### Input format

```
Tmax = 160
Cmax = 9500
City Coordinates:
0 950.5 2500.4 Delhi
1 803.8 2312.3 Jaipur
...
```

### Output

A results report (`outputPS15.txt`) with the best route, total distance and
time, constraint status, GA parameters, a three-method comparison table, and
convergence summary - plus the three PNG figures.

Example run (20-city sample):

| Method            | Distance | Time   | Fitness |
|-------------------|----------|--------|---------|
| Random Search     | varies   | varies | varies  |
| Hill Climbing     | varies   | varies | varies  |
| Genetic Algorithm | varies   | varies | 1.00    |

## Design note (alternate modelling)

Here `time = distance / SPEED`, using `SPEED = 60 km/h`, so time and cost move
together and the trade-off is mild. A truer multi-objective formulation would
give each road its own speed/toll, independent of its length, so the cheapest
tour is not automatically the fastest - producing a genuine time-vs-cost
trade-off.

## Notes

- All debug/test prints are commented out (per the assignment instructions).
- Problem data (`Tmax`, `Cmax`, coordinates) is read from the input file;
  only algorithm parameters are set in code.
