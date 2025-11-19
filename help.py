from math import sqrt

def calculate_distance(point1:tuple, point2:tuple) -> float:
    """
    Calculate the Euclidean distance between two points in 2D space.

    Parameters:
    point1 (tuple): A tuple representing the (x, y) coordinates of the first point.
    point2 (tuple): A tuple representing the (x, y) coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Both points must be 2D coordinates.") 
    
    if not all(isinstance(coord, (int)) and coord>=0 for coord in point1 + point2):
        if not all(isinstance(coord, (int,float)) for coord in point1 + point2):
            raise TypeError("Coordinates must be integers")
        raise TypeError("Coordinates must be positive integers.")

    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_distance_matrix(points:list) -> list:
    """
    Calculate the pairwise Euclidean distances between a list of points.

    Parameters:
    points (list): A list of tuples, each representing the (x, y) coordinates of a point.

    Returns:
    dict: A dictionary where keys are number of the starting point and values are lists of distances to other points.
    """
    if all(not isinstance(point, tuple) or len(point) != 2 for point in points):
        raise ValueError("All points must be 2D coordinates represented as tuples.")
    
    if all(not all(isinstance(coord, (int)) and coord>=0 for coord in point) for point in points):
        if all(not all(isinstance(coord, (int,float)) for coord in point) for point in points):
            raise TypeError("Coordinates must be integers")
        raise TypeError("Coordinates must be positive integers.")

    distance_mat = [[0 for _ in range(len(points))] for _ in range(len(points))]
    for i, point1 in enumerate(points):
        distances = []
        for j, point2 in enumerate(points):
            if i==j:
                distances.append(0.0)
                continue
            distances.append(calculate_distance(point1, point2))
        distance_mat[i] = distances
    return distance_mat

if __name__ == "__main__":
    # Example usage
    pointA = (3, 4)
    pointB = (7, 1)
    distance = calculate_distance(pointA, pointB)
    points = [(0, 0), (3, 4), (6, 8)]
    distance_mat = calculate_distance_matrix(points)
    print("Distance Dictionary:", distance_mat)
    print(f"The distance between {pointA} and {pointB} is {distance}")