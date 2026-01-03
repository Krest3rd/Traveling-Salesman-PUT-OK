# aco_tsp.py
import random
import math
from typing import List, Tuple
from help import calculate_distance, read_points_from_file, calculate_distance_matrix
from naive import naive_tsp
import time
import numpy as np


def tour_length(tour: List[int], distance_mat: np.ndarray) -> float:
    if not tour:
        return np.inf
    idx = np.array(tour + [tour[0]])
    return distance_mat[idx[:-1], idx[1:]].sum()

class AntColony:
    """
    Prosta implementacja ACO dla TSP.
    Parametry:
    - distance_mat: kwadratowa macierz odległości (lista list)
    - n_ants: liczba mrówek na iterację
    - n_iters: liczba iteracji
    - alpha: waga feromonu
    - beta: waga informacji heurystycznej (1/d)
    - rho: współczynnik o ile będzie feromon się osłabiał (0..1) = defultowo 0 dałem
    - q: stała do feromonu
    - init_pheromone: początkowa wartość feromonu (jeśli None to ustawiana automatycznie)
    """
    def __init__(self,
                 distance_mat: np.ndarray,
                 n_ants: int = 20,
                 n_iters: int = 200,
                 alpha: float = 1.0,
                 beta: float = 5.0,
                 rho: float = 0,
                 q: float = 100.0,
                 init_pheromone: float = None,
                 stop_condition: int = 60,
                 verbose: bool = False):
        self.distance_mat = distance_mat
        self.n = len(distance_mat)
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.verbose = verbose
        self.stop_condition = stop_condition
        self.stop_counter = 0
        self.stop_percent = 0.9999


        def build_candidate_lists(distance_mat, K=30):
            n = len(distance_mat)
            order = np.argsort(distance_mat, axis=1)

            # --- Phase 1: Build K-nearest lists with tie handling ---
            k_nearest = []
            for i in range(n):
                neighbors = order[i][order[i] != i]
                d = distance_mat[i][neighbors]

                cut = d[K-1]
                strict = neighbors[d < cut]
                ties = neighbors[d == cut]

                needed = K - len(strict)
                selected = np.concatenate([strict, ties[:needed]])

                k_nearest.append(selected)

            # --- Phase 2: Mutual filtering + refill ---
            mutual = []
            for i in range(n):
                row = []

                for j in k_nearest[i]:
                    if i in k_nearest[j]:
                        row.append(j)

                if len(row) < K:
                    for j in k_nearest[i]:
                        if j not in row:
                            row.append(j)
                            if len(row) == K:
                                break

                mutual.append(np.array(row, dtype=np.int32))

            return k_nearest, mutual

        
        self.candidate_list = [[] for _ in range(self.n)]
        k = math.ceil(6*math.log(self.n))  # liczba najbliższych sąsiadów w liście kandydatów
        self.candidate_list,self.mutual = build_candidate_lists(self.distance_mat, K=k)


        # heurystyczny(zachłanny): eta[i][j] = 1 / dystans 
        d = np.array(distance_mat, dtype=np.float64)
        with np.errstate(divide='ignore'):
            eta = np.where(d > 0, 1.0 / d, 1e9)
        np.fill_diagonal(eta, 0.0)
        self.eta = eta ** self.beta
        self.initialize_pheromone(init_pheromone)

    def initialize_pheromone(self, init_pheromone: float = None):
        # inicjalizacja feromonu
        if init_pheromone is None:
            naive_length = naive_tsp(self.distance_mat,0)[1]
            tau0 = 1.0 / (self.n * naive_length)
            self.pheromone = np.full((self.n, self.n), tau0, dtype=np.float64)
        else:
            self.pheromone = np.full((self.n, self.n), init_pheromone, dtype=np.float64)

    def _select_next(self, current: int, unvisited: set) -> int:
        cand = [c for c in self.candidate_list[current] if c in unvisited]

        if random.random() < 0.01:
            if cand:
                return np.random.choice(cand)  # 1% szansa na wybór zupełnie losowy kandydatów
            else:
                return np.random.choice(list(unvisited))

        if cand:
            choices = np.array(cand)
        else:
            choices = np.array(list(unvisited))

        tau = self.pheromone[current, choices] ** self.alpha
        eta = self.eta[current, choices]
        weights = tau * eta

        if weights.sum() == 0:
            return np.random.choice(choices)

        probs = weights / weights.sum()
        return np.random.choice(choices, p=probs)


    def _construct_solution(self, start: int = None) -> List[int]:
        """Zbuduj jeden tour dla jednej mrówki."""
        if start is None:
            current = random.randrange(self.n)
        else:
            current = start
        tour = [current]
        unvisited = set(range(self.n))
        unvisited.remove(current)
        while unvisited:
            nxt = self._select_next(current, unvisited)
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

# Swap edges until no improvement (slows down significantly way faster convergence)
    def two_opt(self, tour: List[int]) -> Tuple[List[int], float]:
        improved = True
        best_tour = tour.copy()
        best_length = tour_length(best_tour, self.distance_mat)
        positions = [0 for _ in range(len(tour))]

        for i in range(len(tour)):
            positions[tour[i]] = i

        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour)-2):
                city_i = best_tour[i]
                # Only consider j among k nearest neighbors of city_i
                for neighbor in self.mutual[city_i]:
                    j = positions[neighbor]
                    if j - i == 1 or j <= i or j >= len(best_tour) - 1 or j == tour[-1]:  # skip adjacent or invalid
                        continue
                    a, b = best_tour[i-1], best_tour[i]
                    c, d = best_tour[j], best_tour[j+1]
                    delta = (self.distance_mat[a][c] + self.distance_mat[b][d]) - (self.distance_mat[a][b] + self.distance_mat[c][d])
                    if delta < 0:
                        best_tour[i:j+1] = list(reversed(best_tour[i:j+1]))
                        for idx in range(i, j+1): 
                            positions[best_tour[idx]] = idx
                        best_length += delta
                        improved = True

        return best_tour, best_length

    def _update_pheromones(self, all_tours: List[List[int]], all_lengths: List[float]):
        """Osłabianie i zwiększanie feromonu (standardowy mechanizm)."""
        # Evaporation
        self.pheromone *= (1.0 - self.rho)

        # Build deposit matrix
        deposit_matrix = np.zeros_like(self.pheromone)
        for tour, length in zip(all_tours, all_lengths):
            if length <= 0: 
                continue
            deposit = self.q / length
            idx = np.array(tour + [tour[0]])
            edges = np.stack([idx[:-1], idx[1:]], axis=1)
            # Max value to avoid too high pheromone concentration
            deposit_matrix[edges[:,0], edges[:,1]] = np.minimum(deposit_matrix[edges[:,0], edges[:,1]] + deposit, deposit*5)
            deposit_matrix[edges[:,1], edges[:,0]] = np.minimum(deposit_matrix[edges[:,1], edges[:,0]] + deposit, deposit*5)  # symmetry

        self.pheromone += deposit_matrix
        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)

    def run(self) -> Tuple[List[int], float]:
        """Uruchom ACO. Zwraca najlepszy tour i jego długość. Opcjonalnie hist. najlepszych długości."""
                
        if self.verbose:
            print("Starting ACO with parameters:")
            print(f" n_ants={self.n_ants}, n_iters={self.n_iters}, alpha={self.alpha}, beta={self.beta}, rho={self.rho}, q={self.q}")
        best_tour = None
        best_length = float('inf')
        old_best_length = best_length

        k = 3
        if len(self.distance_mat) <= 300:
            x = self.n_ants #all ants
        else:
            x = self.n_ants // 10  # only top 10% ants deposit pheromone
        for iteration in range(0, self.n_iters):
            all_tours = []
            all_lengths = []
            for ant in range(self.n_ants):
                tour = self._construct_solution()
                length = tour_length(tour, self.distance_mat)
                all_tours.append(tour)
                all_lengths.append(length)

            sorted_lengths, sorted_tours = zip(*sorted(zip(all_lengths, all_tours)))
            for i in range(x):  # only top 10% ants deposit pheromone
                tour,length = self.two_opt(sorted_tours[i])
                if length < best_length:
                    best_length = length
                    best_tour = tour
                all_tours[i] = tour
                all_lengths[i] = length

            # sprawdzenie warunku stopu
            if best_length > old_best_length*self.stop_percent:
                self.stop_counter +=1
                self.rho = min(self.rho + 0.01,0.35) # More evaporation when no improvement (explore)
            else:
                self.stop_counter = max(0,self.stop_counter-5) 
                self.rho = max(self.rho - 0.02,0.20) # Less evaporation when improving (exploit)

            old_best_length = min(best_length, old_best_length)
            if self.stop_counter >= self.stop_condition:
                if self.verbose:
                    print(f"Stopping early at iteration {iteration+1} due to no improvement.")
                break

            # dynamiczne ustawianie tau_min i tau_max
            self.tau_max = 1.0 / (self.rho * old_best_length)
            p_best = 0.2
            n = self.n
            pb = p_best ** (1.0 / n)
            self.tau_min = self.tau_max * (1 - pb) / ((n / 2.0 - 1.0) * pb)


            best_tour_idx = all_lengths.index(min(all_lengths[:x:]))
            # aktualizacja feromonów
            self._update_pheromones([all_tours[best_tour_idx]], [all_lengths[best_tour_idx]])
            
            # print("best_length:", best_length, "for iteration:", iteration+1)

            if self.verbose and ((iteration+1) % max(1, self.n_iters//100) == 0 or iteration == 0):
                print(f"Iter {iteration+1}/{self.n_iters}  best_length={old_best_length:.4f}")

        return best_tour + [best_tour[0]], old_best_length

# ======= Example usage (main) =======
if __name__ == "__main__":
    # --- Dla testu: wczytanie z pliku ---
    filename = "./instances/bier127.txt"
    points = read_points_from_file(filename)
    distance_mat = calculate_distance_matrix(points)
    params = {'alpha': 2.0, 'beta': 3.0, 'rho': 0.25, 'q': 272.38618498509874}
    aco = AntColony(distance_mat,
                    n_ants=10,
                    n_iters=500,
                    **params,
                    stop_condition=50,
                    verbose=True)

    # print(tour_length([4, 49, 114, 12, 119, 9, 2, 99, 63, 57, 90, 60, 89, 115, 59, 61, 58, 66, 72, 73, 67, 70, 69, 68, 74, 75, 77, 116, 83, 80, 125, 81, 82, 100, 101, 62, 118, 95, 108, 87, 86, 85, 84, 109, 103, 124, 88, 91, 98, 64, 112, 65, 54, 46, 48, 52, 47, 117, 45, 93, 111, 110, 106, 126, 92, 94, 122, 96, 97, 31, 28, 27, 121, 32, 24, 25, 37, 38, 41, 33, 42, 39, 34, 35, 36, 40, 13, 11, 29, 26, 30, 79, 78, 76, 17, 20, 16, 19, 107, 14, 105, 5, 23, 22, 3, 21, 18, 71, 7, 8, 10, 113, 104, 6, 0, 15, 1, 50, 43, 102, 44, 53, 56, 120, 55, 123, 51, 4],distance_mat))
    start_time = time.perf_counter()
    best_tour, best_len = aco.run()
    end_time = time.perf_counter()
    print("Best tour:", best_tour)
    print("Best length:", best_len)
    print(f"Computation time: {end_time - start_time:.6f} seconds")
    with open("aco_log.txt", "+a") as f:
            f.write(f"ACO run with n_ants={aco.n_ants}, n_iters={aco.n_iters}, alpha={aco.alpha}, beta={aco.beta}, rho={aco.rho}, q={aco.q},\n best_length={best_len:.4f},\n time={end_time - start_time:.6f} seconds,\n filename={filename}\n")


