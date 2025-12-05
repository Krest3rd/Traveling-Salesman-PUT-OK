import glob
from help import calculate_distance, read_points_from_file, calculate_distance_matrix
import random
import time
from ant_colony import AntColony
from naive import naive_tsp
import numpy as np


min_overall = float('inf')
while True:
    ro = random.uniform(0.5, 0.9)
    alfa = random.uniform(0.5, 3.0)
    bet = random.uniform(2.0, 6.0)
    qu = random.uniform(50.0, 300.0)
    for file in glob.glob("./instances/berlin52.txt"):
        print('---------------------------------------------------------------------')
        print(f"Processing file: {file}")
        points = read_points_from_file(file)
        distance_mat = calculate_distance_matrix(points)
        
        alpha=random.uniform(0.5,2.0) 
        beta=random.uniform(2.0,6.0)
        rho=random.uniform(0.7,0.9)
        q=np.mean(distance_mat[distance_mat>0])*random.uniform(0.01,0.10)

        overall = 0
        overall_time = 0.0

        for _ in range (10):
            aco = AntColony(distance_mat,
            n_ants = 200,
            n_iters = 500,
            alpha = alpha,
            beta = beta,
            rho = rho,
            q = q,
            verbose = True
            )
            start_time = time.perf_counter()
            best_tour, best_len = aco.run(return_history=False)
            print("Best tour:", best_tour)
            print("Best length:", best_len)
            end_time = time.perf_counter()
            overall += best_len
            overall_time += end_time - start_time

        avg_len = overall / 10
        avg_time = overall_time / 10
        print("Avg length:", avg_len)
        print(f"Avg computation time: {avg_time:.6f} seconds")
        with open("aco_log.txt", "+a") as f:
            f.write(f"ACO run with n_ants={aco.n_ants}, n_iters={aco.n_iters}, alpha={alfa}, beta={bet}, rho={ro}, q={qu},\n best_length={avg_len:.4f},\n time={avg_time:.6f} seconds,\n filename={file}\n")
    if avg_len < min_overall:
        min_overall = avg_len
        with open("aco_log.txt", "+a") as f:
            f.write(f"New best overall(avg): {min_overall} with average time {overall_time/len(glob.glob('./instances/*.txt')):.6f} seconds\n")