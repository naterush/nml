from collections import defaultdict
import math

def distance_between_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def length(p1):
    return distance_between_points((0, 0), p1)

def get_points_closest_to_point(entire_points, center_points):
    closest = defaultdict(set)
    for point in entire_points:
        closest_center_point = min(*center_points, key=lambda center: distance_between_points(point, center))
        closest[closest_center_point].add(point)
    return closest

def get_kmeans_point(points, k=2, num_iterations=10000, alpha=.01):
    # hardcode 2 for now
    center_points = {
        p for i, p in enumerate(points) if i < k
    }

    for iteration_num in range(num_iterations):
        closest_points_map = get_points_closest_to_point(points, center_points)
        new_center_points = set()
        for center_point, closest_points in closest_points_map.items():
            new_vector = (0, 0)
            for closest_point in closest_points:
                new_vector_update = (center_point[0] - closest_point[0], center_point[1] - closest_point[1])
                if length(new_vector_update) == 0:
                    continue
                new_vector_unit = new_vector_update[0] / length(new_vector_update), new_vector_update[1] / length(new_vector_update)
                new_vector = new_vector[0] + new_vector_unit[0], new_vector[1] + new_vector_unit[1]

            new_center_points.add(
                (center_point[0] - alpha * new_vector[0], center_point[1] - alpha * new_vector[1])
            )
        center_points = new_center_points
        print(f"Iteration {iteration_num}, center points, {center_points}")

    return center_points



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

    kmeans_points = get_kmeans_point(points, k=2)
    print(get_points_closest_to_point(points, kmeans_points))
    

    print(kmeans_points)

main()

# TODO:
# Rather than just updating it in the direction of the points closest, you 
# could instead update it to the middle of these points. This is effectively
# like setting the "max" learning rate. You go in the same direction, but it
# seems like this would converge must faster. Likely, it has really similar 
# convergence properties, it just takes much less time. 

# Also, I should figure out how to factor this to work with numpy arrays, but
# ya know, a fellow can't be pressed too much.
