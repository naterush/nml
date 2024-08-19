import math

def distance_between_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_k_nearest_neighbors(center, points, k=2):

    distance_from_center = {}
    distances = []
    for point in points:
        distance = distance_between_points(center, point)
        distances.append(distance)
        distance_from_center[point] = distance

    distances.sort()
    top = distances[k - 1]
    return list(p for (p, d) in distance_from_center.items() if d <= top)

def main():

    points = {
        # Center of 2, 2
        (1, 2),
        (3, 2),
        (2, 1), 
        (2, 3),

        # Center of 10, 10
        (10, 10),
        (10, 10),
        (10, 10), 
        (10, 10)
    }

    neighbors = get_k_nearest_neighbors((0, 0), points, k=2)
    print(neighbors)

main()