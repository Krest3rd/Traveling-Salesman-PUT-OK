from help import calculate_distance
import time

def read_points_from_file(filename: str) -> list[tuple]:
    """
    Read 2D points from a file.

    File format:
    n               # number of points
    id x y          # each point with its index and coordinates

    Returns:
    list of (x, y) tuples
    """
    points = []
    with open(filename, 'r', encoding='utf-8') as file:
        n = int(file.readline().strip())  # number of points
        for _ in range(n):
            line = file.readline().strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid line format: '{line}'")
            _, x, y = parts
            points.append((int(x), int(y)))
    return points


def closest_pair(start: tuple,points: list[tuple]) -> tuple:
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

    for point in points:
        distance = calculate_distance(start, point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point, min_distance


def naive_tsp(points: list[tuple]) -> tuple:
    """
    Solve the Traveling Salesman Problem (TSP) using a naive nearest neighbor approach.

    Parameters:
    points (list): A list of tuples, each representing the (x, y) coordinates of a point.

    Returns:
    list: The order of points representing the path taken.
    float: The total distance traveled.
    """
    if not isinstance(points, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in points):
        raise TypeError("Input must be a list of 2D coordinate tuples.")
    if not all(isinstance(coord, (int)) and coord>=0 for point in points for coord in point):
        raise TypeError("All coordinates must be positive integers")
    
    if not points:
        return [], 0.0

    unvisited = points[:]
    path = []
    total_distance = 0.0

    # Start from the first point
    current_point = unvisited.pop(0)
    path.append(current_point)

    while unvisited:
        next_point, distance = closest_pair(current_point, unvisited)
        path.append(next_point)
        total_distance += distance
        unvisited.remove(next_point)
        current_point = next_point

    # Return to the starting point to complete the cycle
    return_to_start_distance = calculate_distance(current_point, path[0])
    total_distance += return_to_start_distance
    path.append(path[0])  # Complete the cycle

    return path, total_distance


if __name__ == "__main__":
    # Example usage
    filename = "Instancja_TSP.txt" 
    start_time = time.perf_counter()
    points = read_points_from_file(filename)
    path, total_distance = naive_tsp(points)
    end_time = time.perf_counter()
    print(f"Computation time: {end_time - start_time:.6f} seconds")
    print(f"Path taken: {path}")
    print(f"Total distance traveled: {total_distance:.2f}")