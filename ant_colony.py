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
        self.stop_counter = 0
        self.stop_percent = 0.90



        # heurystyczny(zachłanny): eta[i][j] = 1 / dystans 
        d = np.array(distance_mat, dtype=np.float64)
        with np.errstate(divide='ignore'):
            eta = np.where(d > 0, 1.0 / d, 1e9)
        np.fill_diagonal(eta, 0.0)
        self.eta = eta ** self.beta

        # inicjalizacja feromonu
        if init_pheromone is None:
            naive_length = naive_tsp(self.distance_mat,0)[1]
            tau0 = 1.0 / (self.n * naive_length)
            self.pheromone = np.full((self.n, self.n), tau0, dtype=np.float64)
        else:
            self.pheromone = np.full((self.n, self.n), init_pheromone, dtype=np.float64)

    def _select_next(self, current: int, unvisited: set) -> int:
        """Losowy wybór następnego miasta zgodnie z regułą ACO.
         Zwraca indeks następnego miasta.
         Jest to losowe ale można by to jakoś zachłannym jeszcze dla denom == 0
        """
        unvisited = np.array(list(unvisited))
        tau = self.pheromone[current, unvisited] ** self.alpha
        eta = self.eta[current, unvisited]
        weights = tau * eta
        if weights.sum() == 0:
            return np.random.choice(unvisited)
        probs = weights / weights.sum()
        return np.random.choice(unvisited, p=probs)

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
            deposit_matrix[edges[:,0], edges[:,1]] += deposit
            deposit_matrix[edges[:,1], edges[:,0]] += deposit  # symmetry

        self.pheromone += deposit_matrix
        np.clip(self.pheromone, 1e-16, None, out=self.pheromone)

    def run(self, return_history: bool = False) -> Tuple[List[int], float]:
        """Uruchom ACO. Zwraca najlepszy tour i jego długość. Opcjonalnie hist. najlepszych długości."""
                
        if self.verbose:
            print("Starting ACO with parameters:")
            print(f" n_ants={self.n_ants}, n_iters={self.n_iters}, alpha={self.alpha}, beta={self.beta}, rho={self.rho}, q={self.q}")
        best_tour = None
        best_length = float('inf')
        old_best_length = best_length
        history = []

        for iteration in range(1, self.n_iters + 1):
            all_tours = []
            all_lengths = []
            for ant in range(self.n_ants):
                tour = self._construct_solution()
                length = tour_length(tour, self.distance_mat)
                all_tours.append(tour)
                all_lengths.append(length)
                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()
            # sprawdzenie warunku stopu
            imp = (old_best_length - best_length)/old_best_length if old_best_length != float('inf') else 1.0
            clac = old_best_length*self.stop_percent
            if best_length > old_best_length*self.stop_percent:
                self.stop_counter +=1
                self.rho = min(self.rho + 0.05,0.9) # More evaporation when no improvement (explore)
            else:
                self.stop_counter = max(0,self.stop_counter-5) 
                self.rho = max(self.rho - 0.02,0.1) # Less evaporation when improving (exploit)

            old_best_length = min(best_length, old_best_length)
            if self.stop_counter >= 50:
                if self.verbose:
                    print(f"Stopping early at iteration {iteration} due to no improvement.")
                break

            # aktualizacja feromonów
            self._update_pheromones(all_tours, all_lengths)

            history.append(old_best_length)
            if self.verbose and (iteration % max(1, self.n_iters//10) == 0 or iteration == 1):
                print(f"Iter {iteration}/{self.n_iters}  best_length={best_length:.4f}")

        if return_history:
            return best_tour + [best_tour[0]], old_best_length, history
        else:
            return best_tour + [best_tour[0]], old_best_length

# ======= Example usage (main) =======
if __name__ == "__main__":
    # --- Dla testu: wczytanie z pliku ---
    filename = "./instances/Instancja_TSP.txt"
    points = read_points_from_file(filename)
    distance_mat = calculate_distance_matrix(points)

    aco = AntColony(distance_mat,
                    n_ants=30,
                    n_iters=300,
                    alpha=1.0,
                    beta=7.0,
                    rho=0.5,
                    q=150.0,

                    verbose=True)

    start_time = time.perf_counter()
    best_tour, best_len = aco.run(return_history=False)
    end_time = time.perf_counter()
    print("Best tour:", best_tour)
    print("Best length:", best_len)
    print(f"Computation time: {end_time - start_time:.6f} seconds")
    with open("aco_log.txt", "+a") as f:
            f.write(f"ACO run with n_ants={aco.n_ants}, n_iters={aco.n_iters}, alpha={aco.alpha}, beta={aco.beta}, rho={aco.rho}, q={aco.q},\n best_length={best_len:.4f},\n time={end_time - start_time:.6f} seconds,\n filename={filename}\n")


