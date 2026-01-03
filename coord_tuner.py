import numpy as np
import random
import time
from help import read_points_from_file, calculate_distance_matrix
from ant_colony import AntColony


# ============================================================
# 1. Parameter search helpers
# ============================================================

def evaluate_params(distance_mat, alpha, beta, rho, q, runs=10):
    """Run ACO several times and return average tour length."""
    total = 0.0
    best = float('inf')
    results = []

    for i in range(runs):
        aco = AntColony(
            distance_mat,
            n_ants=100,
            n_iters=500,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
            stop_condition=30,
            verbose=False
        )
        print(f"    Run {i+1}/{runs} with params: alpha={alpha:.4f}, beta={beta:.4f}, rho={rho:.4f}, q={q:.4f}")
        start_time = time.perf_counter()
        _, length = aco.run()
        end_time = time.perf_counter()
        print(f"      Best length: {length:.4f} (time: {end_time - start_time:.4f} seconds)")
        total += length
        results.append(length)

        best = min(best, length)

    avg = total / runs
    median = np.median(results)
    std_div = np.std(results)
    return avg, best, median, std_div


def search_1d(distance_mat, param_name, current_params, search_range, steps=5):
    """Optimize a single parameter while keeping others fixed."""
    best_params = current_params.copy()
    # best_avg = float('inf')
    best_median = float('inf')
    best_avg = float('inf')
    best_len = float('inf')

    low, high = search_range
    values = np.linspace(low, high, steps)

    for v in values:
        test_params = best_params.copy()
        test_params[param_name] = v

        avg, best, median, std_div = evaluate_params(distance_mat, **test_params)

        print(f"  Testing {param_name}={v:.4f} → avg={avg:.4f}, median={median:.4f}, best={best:.4f}, std={std_div:.4f}")

        if std_div/median > 0.01:
            print("    Warning: High variability in results (std/median > 1%) skipping update.")
            continue

        if median < best_median:
            best_median = median
            best_params = test_params.copy()
            best_avg = avg

        if best < best_len:
            best_len = best
            with open("aco_log_coordinate_descent.txt", "+a") as f:
                f.write(f"New best length: {best_len:.4f} with params: {test_params}, avg={avg:.4f} and median={median:.4f}\n")


    print(f"  → Best {param_name} = {best_params[param_name]:.4f} (avg={best_avg:.4f}, median={best_median:.4f})")
    return best_params, best_avg, best_median


# ============================================================
# 2. Full coordinate descent loop
# ============================================================

def coordinate_descent(distance_mat, max_cycles=10):
    # Initial parameter guesses
    params = {
        "alpha": random.uniform(0.5, 2.0),
        "beta": random.uniform(1.0, 3.0),
        "rho": random.uniform(0.2, 0.3),
        "q": np.mean(distance_mat[distance_mat > 0]) * random.uniform(0.01, 0.10)
    }

    print("Initial parameters:", params)

    # Search ranges for each parameter
    ranges = {
        "alpha": (0.5, 2.0),
        "beta": (1.0, 3.0),
        "rho": (0.2, 0.3),
        "q": (0.01 * np.mean(distance_mat[distance_mat > 0]),
              0.10 * np.mean(distance_mat[distance_mat > 0]))
    }

    prev_best = float('inf')

    for cycle in range(max_cycles):
        print(f"\n====================")
        print(f"Cycle {cycle+1}")
        print("====================")

        for param in np.random.permutation(["alpha", "beta", "rho", "q"]):
            print(f"\nOptimizing {param}...")
            params, best_avg, best_med = search_1d(distance_mat, param, params, ranges[param])

        print(f"\nCycle {cycle+1} best avg = {best_avg:.4f}, best median = {best_med:.4f}")

        # Stopping condition: no improvement
        if best_med >= prev_best - 1e-6:
            print("\nNo improvement detected. Stopping coordinate descent.")
            break
        
        with open("aco_log_coordinate_descent.txt", "+a") as f:
            f.write(f"Cycle {cycle+1}: params={params}, best_avg={best_avg:.4f}, best_median={best_med:.4f}\n")
        prev_best = best_med

    print("\nFinal optimized parameters:", params)
    return params


# ============================================================
# 3. Main entry point
# ============================================================

if __name__ == "__main__":
    filename = "./instances/bier127.txt"
    points = read_points_from_file(filename)
    distance_mat = calculate_distance_matrix(points)

    best_params = coordinate_descent(distance_mat)
    print("\nBest parameters found:", best_params)
