import glob
from help import calculate_distance, read_points_from_file, calculate_distance_matrix
import random
import time
from ant_colony import AntColony



while True:
    min_overall = float('inf')
    overall = 0
    overall_time = 0.0
    for file in glob.glob("./instances/*.txt"):
        print('---------------------------------------------------------------------')
        print(f"Processing file: {file}")
        points = read_points_from_file(file)
        distance_mat = calculate_distance_matrix(points)
        aco = AntColony(distance_mat,
            n_ants = 100,
            n_iters = 500,
            alpha = random.uniform(0.5, 3.0),
            beta = random.uniform(1.0, 10.0),
            rho = random.uniform(0.1, 0.9),
            q = random.uniform(50.0, 300.0),
            verbose = True
        )
        start_time = time.perf_counter()
        best_tour, best_len = aco.run(return_history=False)
        end_time = time.perf_counter()
        overall += best_len
        overall_time += end_time - start_time
        print("Best tour:", best_tour)
        print("Best length:", best_len)
        print(f"Computation time: {end_time - start_time:.6f} seconds")
        with open("aco_log.txt", "+a") as f:
            f.write(f"ACO run with n_ants={aco.n_ants}, n_iters={aco.n_iters}, alpha={aco.alpha}, beta={aco.beta}, rho={aco.rho}, q={aco.q},\n best_length={best_len:.4f},\n time={end_time - start_time:.6f} seconds,\n filename={file}\n")
    if overall < min_overall:
        min_overall = overall
        with open("aco_log.txt", "+a") as f:
            f.write(f"New best overall: {min_overall} with average time {overall_time/len(glob.glob('./instances/*.txt')):.6f} seconds\n")