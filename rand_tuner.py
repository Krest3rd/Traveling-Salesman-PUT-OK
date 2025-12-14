import glob
from help import read_points_from_file, calculate_distance_matrix
import random
import time
from ant_colony import AntColony
import numpy as np
import cProfile

minavg_overall = float('inf')
min_overall = float('inf')
while True:

    file = "./instances/tsp1000.txt"
    points = read_points_from_file(file)
    distance_mat = calculate_distance_matrix(points)
    alpha=random.uniform(0.5,2.0) 
    beta=random.uniform(2.0,6.0)
    rho=random.uniform(0.7,0.9)
    q=np.mean(distance_mat[distance_mat>0])*random.uniform(0.01,0.10)

    for file in glob.glob("./instances/tsp1000.txt"):
        print('---------------------------------------------------------------------')
        print(f"Processing file: {file}")
        points = read_points_from_file(file)
        distance_mat = calculate_distance_matrix(points)

        overall = 0
        overall_time = 0.0
        overall_best = float('inf')
        for i in range (5):
            print(f"Run {i+1}/5")
            aco = AntColony(distance_mat,
            n_ants = 50,
            n_iters = 500,
            alpha = alpha,
            beta = beta,
            rho = rho,
            q = q,
            stop_condition=10,
            verbose = True
            )
            start_time = time.perf_counter()
            # best_tour, best_len = aco.run(return_history=False)
            cProfile.run('best_tour, best_len = aco.run()')
            print("Best tour:", best_tour)
            print("Best length:", best_len)
            end_time = time.perf_counter()
            if best_len < overall_best:
                overall_best = best_len
            overall += best_len
            overall_time += end_time - start_time
            print(f"Computation time: {end_time - start_time:.6f} seconds\n")

    if overall_best <= min_overall:
        min_overall = overall_best
        with open("aco_log.txt", "+a") as f:
            f.write(f"New best overall: {min_overall}\n")

    avg_len = overall / 5
    avg_time = overall_time / 5
    print("Avg length:", avg_len)
    print(f"Avg computation time: {avg_time:.6f} seconds")
    with open("aco_log.txt", "+a") as f:
        f.write(f"ACO run with n_ants={aco.n_ants}, n_iters={aco.n_iters}, alpha={alpha}, beta={beta}, rho={rho}, q={q}, stop_cond={aco.stop_condition}\n best_length={overall_best:.4f},\n avg_len={avg_len}\n time={avg_time:.6f} seconds,\n filename={file}\n")
    if avg_len <= minavg_overall:
        minavg_overall = avg_len
        with open("aco_log.txt", "+a") as f:
            f.write(f"New best overall(avg): {minavg_overall} with average time {overall_time/len(glob.glob('./instances/*.txt')):.6f} seconds\n")
    