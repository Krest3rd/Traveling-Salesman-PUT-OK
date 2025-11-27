from help import calculate_distance, read_points_from_file, calculate_distance_matrix
import time
import numpy as np


def closest_pair(start: int, distance_mat: list, unvisited: set) -> tuple:
    """
    Find the closest point to a given starting point from a list of points.

    Parameters:
    start (tuple): A tuple representing the (x, y) coordinates of the starting point.
    points (list): A list of tuples, each representing the (x, y) coordinates of a point.

    Returns:
    tuple: The point from the list that is closest to the starting point.
    float: The Euclidean distance to the closest point.
    """
    min_distance = float('inf')
    closest_point = None

    for i in unvisited:
        if i == start:
            continue
        distance = distance_mat[start][i]
        if distance < min_distance:
            min_distance = distance
            closest_point = i

    return closest_point, min_distance


def naive_tsp(distance_mat: np.ndarray, start:int) -> tuple:
    """
    Solve the Traveling Salesman Problem (TSP) using a naive nearest neighbor approach.

    Parameters:
    distances (list): A list of distances between points ([start][end]).
    start (int): The starting point index.

    Returns:
    list: The order of points representing the path taken.
    float: The total distance traveled.
    """


    n = distance_mat.shape[0]
    total_distance = 0.0
    path = [start]

    # Boolean mask for unvisited nodes
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False
    current = start

    while unvisited.any():
        # Extract distances from current to all unvisited
        dists = np.where(unvisited, distance_mat[current], np.inf)

        # Pick the closest unvisited city
        next_point = np.argmin(dists)
        distance = dists[next_point]

        path.append(next_point)
        total_distance += distance
        unvisited[next_point] = False
        current = next_point

    # Return to start
    return_to_start = distance_mat[current, start]
    total_distance += return_to_start
    path.append(start)

    return path, total_distance


if __name__ == "__main__":
    # Example usage
    filename = "instancja_TSP.txt"
    points = read_points_from_file(filename)
    mat = calculate_distance_matrix(points)
    start_time = time.perf_counter()
    path, total_distance = naive_tsp(mat,0)
    end_time = time.perf_counter()
    print(f"Computation time: {end_time - start_time:.6f} seconds")
    print(f"Path taken: {path}")
    print(f"Total distance traveled: {total_distance:.2f}")