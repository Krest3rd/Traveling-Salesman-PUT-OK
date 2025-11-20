from help import calculate_distance
import time


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


def naive_tsp(distances: list[list[int]], start:int) -> tuple:
    """
    Solve the Traveling Salesman Problem (TSP) using a naive nearest neighbor approach.

    Parameters:
    distances (list): A list of distances between points ([start][end]).
    start (int): The starting point index.

    Returns:
    list: The order of points representing the path taken.
    float: The total distance traveled.
    """
    if not isinstance(distances, list) or not all(isinstance(row, list) for row in distances):
        raise TypeError("Distances must be a 2D list.")

    if any(len(row) != len(distances) for row in distances):
        raise ValueError("Distance matrix must be square.")


    total_distance = 0.0
    path = []

    # Start from the first point
    current_point = start
    path.append(current_point)
    unvisited = set(range(len(distances)))
    unvisited.remove(current_point)

    while distances:
        next_point, distance = closest_pair(current_point, distances, unvisited)
        path.append(next_point)
        total_distance += distance
        unvisited.remove(next_point)
        current_point = next_point

    # Return to the starting point to complete the cycle
    return_to_start_distance = distances[current_point, path[0]]
    total_distance += return_to_start_distance
    path.append(path[0])  # Complete the cycle

    return path, total_distance


if __name__ == "__main__":
    # Example usage
    filename = "tsp250.txt" 
    start_time = time.perf_counter()
    path, total_distance = naive_tsp(points)
    end_time = time.perf_counter()
    print(f"Computation time: {end_time - start_time:.6f} seconds")
    print(f"Path taken: {path}")
    print(f"Total distance traveled: {total_distance:.2f}")