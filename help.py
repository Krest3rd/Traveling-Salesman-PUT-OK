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
    
    if not all(isinstance(coord, (int)) and coord>0 for coord in point1 + point2):
        if not all(isinstance(coord, (int,float)) for coord in point1 + point2):
            raise TypeError("Coordinates must be integers")
        raise TypeError("Coordinates must be positive integers.")

    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

if __name__ == "__main__":
    # Example usage
    pointA = (3, 4)
    pointB = (7, 1)
    distance = calculate_distance(pointA, pointB)
    print(f"The distance between {pointA} and {pointB} is {distance}")