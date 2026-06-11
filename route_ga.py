"""
=====================================================================
 ACI Assignment 1 - PS15 : Route Optimization with a Genetic Algorithm
=====================================================================
 Problem (plain terms):
   Visit all N cities exactly once and return to the origin (a closed
   tour), keeping total travel TIME <= Tmax and total COST <= Cmax,
   while making the tour as short / fast as possible.

 Required techniques (from the brief):
   a) Permutation encoding   -> a route is an ordering of city ids
   b) Ordered crossover (OX) -> breeds two routes into a valid child
   c) Swap mutation          -> randomly swaps two cities in a route
   d) Rank selection         -> picks parents by rank, not raw score

 We also compare the GA against two simpler baselines:
   - Random search
   - Hill climbing (mirrors the structure of the instructor's template)

 Design note (for the report's "alternate modelling" section):
   Here TIME is derived from distance (time = distance / SPEED), so the
   two objectives move together and the trade-off is mild. A truer
   multi-objective version would give each road its OWN speed/toll,
   independent of its length, so the cheapest tour is not automatically
   the fastest -- creating a real time-vs-cost trade-off.
=====================================================================
"""

import sys
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend: save figures, no GUI needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MODEL PARAMETERS (tunable constants - NOT problem data).
# Problem data (Tmax, Cmax, city coordinates) is read from the input file;
# these are algorithm settings and are allowed to be set here.
# ---------------------------------------------------------------------------
SPEED       = 60.0    # km per hour -> time = distance / SPEED
W_DIST      = 1.0     # weight on distance in the combined objective
W_TIME      = 1.0     # weight on time in the combined objective
PENALTY_BIG = 1000.0  # multiplier used to punish constraint violations

POP_SIZE    = 120     # number of routes kept alive each generation
GENERATIONS = 500     # how many generations the GA runs
MUTATION_RT = 0.20    # probability a child undergoes a swap mutation
ELITES      = 2       # best routes copied unchanged into the next generation
RANDOM_ITER = 5000    # iterations for the random-search baseline
HILL_ITER   = 5000    # iterations for the hill-climbing baseline


# ===========================================================================
# 1. BOUNDED POPULATION CONTAINER
#    A simple fixed-capacity collection. Its add()/remove() raise clear
#    messages when full / empty, satisfying the brief's requirement that
#    data-structure insert and delete operations report capacity errors.
# ===========================================================================
class Population:
    """A fixed-capacity list of routes (each route is a permutation)."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Population capacity must be a positive integer.")
        self.capacity = capacity
        self.members = []

    def add(self, route):
        """Insert a route; refuses (and reports) when the population is full."""
        if len(self.members) >= self.capacity:
            raise OverflowError("Population is FULL - cannot add another route.")
        self.members.append(route)

    def remove(self):
        """Delete and return the last route; reports when the population is empty."""
        if len(self.members) == 0:
            raise IndexError("Population is EMPTY - nothing to remove.")
        return self.members.pop()

    def __len__(self):
        return len(self.members)


# ===========================================================================
# 2. INPUT / OUTPUT
# ===========================================================================
def read_input(filename):
    """
    Read the problem definition from a text file.

    Expected format (whitespace/'=' tolerant):
        Tmax = 150
        Cmax = 1100
        City Coordinates:
        0 950.5 2500.4 Delhi
        1 803.8 2312.3 Jaipur
        ...
    Returns: (Tmax, Cmax, cities, city_names) where cities is an (N x 2)
             numpy array indexed by city id.
    """
    try:
        with open(filename, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        raise FileNotFoundError("Input file '%s' not found." % filename)

    Tmax = Cmax = None
    coords = {}
    names = {}
    for ln in lines:
        low = ln.lower()
        if low.startswith("tmax"):
            Tmax = float(ln.replace("=", " ").split()[-1])
        elif low.startswith("cmax"):
            Cmax = float(ln.replace("=", " ").split()[-1])
        elif low.startswith("city"):
            continue                      # the "City Coordinates:" header line
        else:
            parts = ln.split()
            if len(parts) >= 3:           # "<id> <x> <y> [city name]"
                cid, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coords[cid] = (x, y)
                names[cid] = " ".join(parts[3:]) if len(parts) > 3 else str(cid)

    # Basic validation / error handling.
    if Tmax is None or Cmax is None:
        raise ValueError("Input file is missing Tmax and/or Cmax.")
    if len(coords) == 0:
        raise ValueError("Input file contains no city coordinates.")
    ids = sorted(coords)
    if ids != list(range(len(ids))):
        raise ValueError("City ids must be contiguous and start at 0.")

    cities = np.array([coords[i] for i in ids], dtype=float)
    city_names = [names[i] for i in ids]
    return Tmax, Cmax, cities, city_names


def generate_input_file(filename, n_cities=20, seed=42,
                        tmax=150.0, cmax=1100.0):
    """
    Task (a): randomly generate N city coordinates and save them in the
    required input format. Used only if no input file is supplied.
    """
    rng = random.Random(seed)
    with open(filename, "w") as f:
        f.write("Tmax = %g\n" % tmax)
        f.write("Cmax = %g\n" % cmax)
        f.write("City Coordinates:\n")
        for i in range(n_cities):
            f.write("%d %d %d\n" % (i, rng.randint(0, 100), rng.randint(0, 100)))


def write_output(filename, ga, baselines, params, conv):
    """Write results to the output file in the brief's reporting format."""
    with open(filename, "w") as f:
        f.write("Genetic Algorithm Results\n")
        f.write("-------------------------\n")
        f.write("Best Route:\n")
        f.write(route_to_str(ga["route"], params.get("city_names")) + "\n")
        f.write("Total Distance (Cost): %.2f\n" % ga["distance"])
        f.write("Total Travel Time: %.2f hours\n\n" % ga["time"])

        f.write("Constraints Status:\n")
        f.write("Time <= Tmax : %s\n" %
                ("SATISFIED" if ga["time"] <= params["Tmax"] else "VIOLATED"))
        f.write("Cost <= Cmax : %s\n\n" %
                ("SATISFIED" if ga["distance"] <= params["Cmax"] else "VIOLATED"))

        f.write("GA Parameters:\n")
        f.write("Population Size = %d\n" % params["pop"])
        f.write("Generations = %d\n" % params["gens"])
        f.write("Crossover = Ordered Crossover (OX)\n")
        f.write("Mutation = Swap Mutation\n")
        f.write("Selection = Rank Selection\n\n")

        if params.get("city_names"):
            f.write("City Index:\n")
            for i, name in enumerate(params["city_names"]):
                f.write("%d = %s\n" % (i, name))
            f.write("\n")

        f.write("Performance Comparison\n")
        f.write("---------------------------------------------------\n")
        f.write("%-18s %-11s %-9s %s\n" % ("Method", "Distance", "Time", "Fitness"))
        f.write("---------------------------------------------------\n")
        for name, r in baselines:
            f.write("%-18s %-11.2f %-9.2f %.2f\n" %
                    (name, r["distance"], r["time"], r["norm_fitness"]))
        f.write("\n")

        f.write("Convergence:\n")
        f.write("Best Distance improved from %.2f -> %.2f\n" %
                (conv["dist0"], conv["distF"]))
        f.write("Best Time improved from %.2f -> %.2f\n\n" %
                (conv["time0"], conv["timeF"]))

        f.write("Visualization Generated:\n")
        f.write("1. Best Route Plot\n")
        f.write("2. Distance Convergence Graph\n")
        f.write("3. Time Convergence Graph\n")


def route_to_str(route, city_names=None):
    """Format a tour starting and ending at origin 0."""
    rot = rotate_to_origin(route)
    loop = rot + [rot[0]]
    if city_names:
        return " -> ".join("%s(%d)" % (city_names[c], c) for c in loop)
    return " -> ".join(str(c) for c in loop)


# ===========================================================================
# 3. DISTANCE / TIME / FITNESS  (the evaluation function)
# ===========================================================================
def build_distance_matrix(cities):
    """Pre-compute the N x N Euclidean distance matrix (done once, for speed)."""
    n = len(cities)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dmat[i, j] = math.dist(cities[i], cities[j])
    return dmat


def route_distance(route, dmat):
    """
    Total tour distance = sum of consecutive legs PLUS the closing leg
    back to the origin (the tour is a closed loop).
    """
    if route is None or len(route) == 0:
        raise ValueError("route_distance received an empty route.")
    total = 0.0
    for i in range(len(route)):
        a = route[i]
        b = route[(i + 1) % len(route)]   # wraps last city back to first
        total += dmat[a, b]
    return total


def route_time(distance):
    """Simple time model: time = distance / SPEED (see design note up top)."""
    return distance / SPEED


def evaluate(route, dmat, Tmax, Cmax):
    """
    Score a route. Returns distance, time, feasibility and a fitness value
    (higher = better). Constraint breaks add a proportional penalty so bad
    routes score poorly but are not deleted from the gene pool.
    """
    dist = route_distance(route, dmat)
    tim  = route_time(dist)

    penalty = 0.0
    if tim > Tmax:
        penalty += PENALTY_BIG * (tim - Tmax)
    if dist > Cmax:
        penalty += PENALTY_BIG * (dist - Cmax)

    combined_clean = W_DIST * dist + W_TIME * tim          # objective only
    combined = combined_clean + penalty                    # objective + penalty
    fitness  = 1.0 / combined            # higher fitness for lower combined cost
    feasible = (tim <= Tmax) and (dist <= Cmax)
    return {"distance": dist, "time": tim, "combined": combined,
            "combined_clean": combined_clean, "fitness": fitness,
            "feasible": feasible, "route": route}


# ===========================================================================
# 4. GA OPERATORS
# ===========================================================================
def init_population(pop_size, n_cities):
    """Create POP_SIZE random permutations of the city ids (Task: encoding)."""
    base = list(range(n_cities))
    pop = []
    for _ in range(pop_size):
        r = base[:]
        random.shuffle(r)
        pop.append(r)
    return pop


def rank_selection(population, fitnesses):
    """
    Rank selection: sort by fitness, give the worst rank 1 and the best
    rank N, then pick a parent with probability proportional to rank.
    Using rank (not raw fitness) stops one lucky route from dominating.
    """
    n = len(population)
    order = np.argsort(fitnesses)         # indices, worst -> best
    ranks = np.empty(n)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    probs = ranks / ranks.sum()
    chosen = np.random.choice(n, p=probs)
    return population[chosen]


def ordered_crossover(parent1, parent2):
    """
    Ordered Crossover (OX): copy a random slice from parent1, then fill the
    remaining slots with parent2's cities in order, skipping any already
    taken. This guarantees the child is still a valid permutation.
    """
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))   # two cut points
    child = [None] * n
    child[a:b + 1] = parent1[a:b + 1]           # keep parent1's middle slice
    taken = set(child[a:b + 1])
    fill = [c for c in parent2 if c not in taken]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def swap_mutation(route, rate):
    """With probability `rate`, swap two random positions (Task: mutation)."""
    new = route[:]
    if random.random() < rate:
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new


def rotate_to_origin(route):
    """Rotate a tour so it begins at city 0 (for tidy display only)."""
    if 0 in route:
        k = route.index(0)
        return route[k:] + route[:k]
    return route[:]


def genetic_algorithm(dmat, n_cities, Tmax, Cmax):
    """
    Main GA loop: evaluate -> select -> crossover -> mutate, repeated for
    GENERATIONS, with elitism so the best routes are never lost.
    Returns the best route's evaluation plus per-generation history for plots.
    """
    population = init_population(POP_SIZE, n_cities)

    best = None
    dist_history, time_history = [], []

    for gen in range(GENERATIONS):
        evals = [evaluate(r, dmat, Tmax, Cmax) for r in population]
        fitnesses = [e["fitness"] for e in evals]

        # Track the best route seen so far.
        gen_best = max(evals, key=lambda e: e["fitness"])
        if best is None or gen_best["fitness"] > best["fitness"]:
            best = gen_best
        dist_history.append(best["distance"])
        time_history.append(best["time"])

        # Build the next generation inside a bounded Population container.
        nextgen = Population(POP_SIZE)
        #Elitism
        elite_routes = [e["route"] for e in
                        sorted(evals, key=lambda e: e["fitness"], reverse=True)[:ELITES]]
        for er in elite_routes:
            nextgen.add(er[:])            # carry the elites over unchanged
        #parent selection and breeding
        while len(nextgen) < POP_SIZE:
            p1 = rank_selection(population, fitnesses)
            p2 = rank_selection(population, fitnesses)
            child = ordered_crossover(p1, p2)
            child = swap_mutation(child, MUTATION_RT)
            nextgen.add(child)

        population = nextgen.members

        # --- debug (commented out before submission, per the brief) ---
        # print("Gen %d  best dist=%.2f  time=%.2f" %
        #       (gen, best["distance"], best["time"]))

    best["dist_history"] = dist_history
    best["time_history"] = time_history
    return best


# ===========================================================================
# 5. BASELINES (for the performance comparison)
# ===========================================================================
def random_search(dmat, n_cities, Tmax, Cmax, iterations):
    """Throw darts: try many random tours, keep the best one found."""
    best = None
    base = list(range(n_cities))
    for _ in range(iterations):
        r = base[:]
        random.shuffle(r)
        e = evaluate(r, dmat, Tmax, Cmax)
        if best is None or e["fitness"] > best["fitness"]:
            best = e
    return best


def hill_climbing(dmat, n_cities, Tmax, Cmax, iterations):
    """
    Greedy local search (same shape as the instructor's template): start
    from a random tour, repeatedly swap two cities, and keep the change
    only if it improves the score. Gets stuck in local optima.
    """
    base = list(range(n_cities))
    random.shuffle(base)
    current = evaluate(base, dmat, Tmax, Cmax)

    for _ in range(iterations):
        neighbour = swap_mutation(current["route"], rate=1.0)   # force one swap
        cand = evaluate(neighbour, dmat, Tmax, Cmax)
        if cand["fitness"] > current["fitness"]:                # accept if better
            current = cand
    return current


# ===========================================================================
# 6. VISUALISATION
# ===========================================================================
def plot_route(cities, route, path, city_names=None):
    """Best-route map: cities as points, the tour drawn as a closed loop."""
    rot = rotate_to_origin(route)
    loop = rot + [rot[0]]
    xs = [cities[c][0] for c in loop]
    ys = [cities[c][1] for c in loop]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, "-o")
    for c in route:
        label = city_names[c] if city_names else str(c)
        plt.annotate(label, (cities[c][0], cities[c][1]), fontsize=8)
    plt.title("Best Route Found (GA)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.savefig(path, bbox_inches="tight"); plt.close()


def plot_convergence(history, ylabel, title, path):
    """Generic convergence plot of a quantity vs generation."""
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(history)), history)
    plt.title(title)
    plt.xlabel("Generation"); plt.ylabel(ylabel)
    plt.savefig(path, bbox_inches="tight"); plt.close()


# ===========================================================================
# 7. MAIN
# ===========================================================================
def main():
    in_file  = sys.argv[1] if len(sys.argv) > 1 else "inputPS15.txt"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "outputPS15.txt"

    # If no input file exists, generate one (Task a) so the program still runs.
    try:
        open(in_file).close()
    except FileNotFoundError:
        print("No input file found; generating a random one at '%s'." % in_file)
        generate_input_file(in_file)

    Tmax, Cmax, cities, city_names = read_input(in_file)
    n = len(cities)
    dmat = build_distance_matrix(cities)

    # --- Run the three methods ---
    ga = genetic_algorithm(dmat, n, Tmax, Cmax)
    rs = random_search(dmat, n, Tmax, Cmax, RANDOM_ITER)
    hc = hill_climbing(dmat, n, Tmax, Cmax, HILL_ITER)

    # --- Normalised fitness for the table (best method = 1.00) ---
    # Based on the objective only (excludes the steering penalty) so the
    # column reads as a clean 0-1 score even for infeasible baselines.
    best_clean = min(rs["combined_clean"], hc["combined_clean"], ga["combined_clean"])
    for r in (rs, hc, ga):
        r["norm_fitness"] = round(best_clean / r["combined_clean"], 2)

    # --- Console report ---
    print("Best Route:", route_to_str(ga["route"], city_names))
    print("Total Distance (Cost): %.2f" % ga["distance"])
    print("Total Travel Time: %.2f hours" % ga["time"])
    print("Time <= Tmax :", "SATISFIED" if ga["time"] <= Tmax else "VIOLATED")
    print("Cost <= Cmax :", "SATISFIED" if ga["distance"] <= Cmax else "VIOLATED")
    print("\n%-18s %-11s %-9s %s" % ("Method", "Distance", "Time", "Fitness"))
    for name, r in [("Random Search", rs), ("Hill Climbing", hc),
                    ("Genetic Algorithm", ga)]:
        print("%-18s %-11.2f %-9.2f %.2f" %
              (name, r["distance"], r["time"], r["norm_fitness"]))

    # --- Plots ---
    plot_route(cities, ga["route"], "best_route.png", city_names)
    plot_convergence(ga["dist_history"], "Best Distance",
                     "Distance Convergence", "distance_convergence.png")
    plot_convergence(ga["time_history"], "Best Time",
                     "Time Convergence", "time_convergence.png")

    # --- Output file ---
    conv = {"dist0": ga["dist_history"][0], "distF": ga["dist_history"][-1],
            "time0": ga["time_history"][0], "timeF": ga["time_history"][-1]}
    params = {"Tmax": Tmax, "Cmax": Cmax, "pop": POP_SIZE,
              "gens": GENERATIONS, "city_names": city_names}
    baselines = [("Random Search", rs), ("Hill Climbing", hc),
                 ("Genetic Algorithm", ga)]
    write_output(out_file, ga, baselines, params, conv)
    print("\nWrote results to '%s' and 3 PNG figures." % out_file)


if __name__ == "__main__":
    main()
