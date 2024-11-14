import numpy as np
import random
from scipy.spatial import ConvexHull, Delaunay
import os
current_script_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(current_script_path)

from utils.closest_point import closest_point_on_mesh_slow, closest_point_on_mesh_fast, build_triangle_centroid_kdtree

random.seed(42)
np.random.seed(42)

def generate_random_convex_polygon(N):
    """Generates a random convex polygon.
    
    Args:
        N (int): number of points in the initial polygon

    Returns:
        hull (ConvexHull): the scipy Convex Hull object defining the convex
            polygon    
    """
    # Generate N random points in 3D space
    points = np.random.rand(N, 3)
    
    # Compute the convex hull of these points
    hull = ConvexHull(points)
    
    # Use only the vertices needed for the convex hull
    points = hull.points[hull.vertices]
    hull = ConvexHull(points)
    
    return hull

def random_point_on_triangle_plane(vertices, triangle_indices_row):
    """Return a random point on the plane of a triangle."""
    A, B, C = vertices[triangle_indices_row]
    u, v = np.random.rand(2)
    if u + v > 1:
        u, v = 1 - u, 1 - v
    return (1 - u - v) * A + u * B + v * C

def random_point_on_triangle_edge(vertices, triangle_indices_row):
    """Return a random point on one of the edges of a triangle."""
    A, B, C = vertices[triangle_indices_row]
    edges = [(A, B), (B, C), (C, A)]
    edge = edges[np.random.choice(3)]
    t = np.random.rand()
    return (1 - t) * edge[0] + t * edge[1]

def get_normal_unit_vector_from_triangle(vertices, triangle_indices, convex_hull):
    """Returns the unit vector normal to the triangle that points outside the convex hull."""
    a, b, c = vertices[triangle_indices]
    ab, ac = b - a, c - a
    normal_vector = np.cross(ab, ac)
    
    # Handle collinear case
    norm = np.linalg.norm(normal_vector)
    if np.isclose(norm, 0):
        raise ValueError("The triangle vertices are collinear; normal vector cannot be defined.")
    
    unit_vec_1 = normal_vector / norm
    unit_vec_2 = -unit_vec_1
    
    offset_distance = 5 # Small offset for testing
    test_point_1 = a + unit_vec_1 * offset_distance
    test_point_2 = a + unit_vec_2 * offset_distance
    
    test_point_1_inside = is_point_inside_hull(test_point_1, convex_hull)
    test_point_2_inside = is_point_inside_hull(test_point_2, convex_hull)

    assert not (test_point_1_inside and test_point_2_inside)
    # the below means that both test points are outside the hull, meaning 
    # we are not sure what point it is closes to, so we skip this point
    if test_point_1_inside == test_point_2_inside:
        return None

    if test_point_1_inside:
        return unit_vec_2
    else:
        return unit_vec_1

def is_point_inside_hull(point, hull):
    delaunay = Delaunay(hull.points)
    return delaunay.find_simplex(point) >= 0

def generate_test_closest_point_test_case(num_vertices):
    convex_hull = generate_random_convex_polygon(num_vertices)
    vertices = convex_hull.points[convex_hull.vertices]
    triangle_indices = convex_hull.simplices

    test_pcd, nearest_points, dist = [], [], []
    for t in triangle_indices:
        
        pt_on_plane = random_point_on_triangle_plane(vertices, t)
        test_pcd.append(pt_on_plane)
        nearest_points.append(pt_on_plane)
        dist.append([0])

        pt_on_edge = random_point_on_triangle_edge(vertices, t)
        test_pcd.append(pt_on_edge)
        nearest_points.append(pt_on_edge)
        dist.append([0])

        vertex_idx = random.randint(0, 2)
        pt_on_vertex = vertices[t[vertex_idx]]
        test_pcd.append(pt_on_vertex)
        nearest_points.append(pt_on_vertex)
        dist.append([0])

        norm_unit_vec = get_normal_unit_vector_from_triangle(vertices, t, convex_hull)
        if norm_unit_vec is not None:
            distance = random.uniform(0, 10)
            test_pcd.append(pt_on_plane + norm_unit_vec * distance)
            nearest_points.append(pt_on_plane)
            dist.append([distance])

            distance = random.uniform(0, 10)
            test_pcd.append(pt_on_edge + norm_unit_vec * distance)
            nearest_points.append(pt_on_edge)
            dist.append([distance])

            distance = random.uniform(0, 10)
            test_pcd.append(pt_on_vertex + norm_unit_vec * distance)
            nearest_points.append(pt_on_vertex)
            dist.append([distance])

    mesh = {"V": vertices, "i": triangle_indices}

    return np.array(test_pcd), mesh, np.array(nearest_points), np.array(dist)

def test_closest_point_algorithm_slow():
    """Generates a custom 3D, convex mesh and then generates a random pcd
    with ground truth closest points on the mesh and distances. The predicted
    closest points and distances are asserted to be identical to the ground
    truth."""
    test_pcd, mesh, nearest_points, distances = generate_test_closest_point_test_case(1500)

    pred_closest = list()
    pred_dists = list()

    for idx, p in enumerate(test_pcd):
        closest, dist = closest_point_on_mesh_slow(p, mesh)
        pred_closest.append(closest)
        pred_dists.append([dist])

    pred_closest = np.array(pred_closest)
    pred_dists = np.array(pred_dists)

    with open(f"{CUR_DIR}/test_closest_point_results_slow.txt", "w") as file:
        mse = np.average((pred_closest-nearest_points)**2)
        file.write(f"MSE: {mse}\n")
        file.writelines([f"{x} - {y}\n" for x, y in zip(pred_closest, nearest_points)])

    assert np.isclose(np.average((pred_closest-nearest_points)**2) + np.average((pred_dists-distances)**2), 0)
 

def test_closest_point_algorithm_fast():
    """Generates a custom 3D, convex mesh and then generates a random pcd
    with ground truth closest points on the mesh and distances. The predicted
    closest points and distances are asserted to be identical to the ground
    truth."""
    test_pcd, mesh, nearest_points, distances = generate_test_closest_point_test_case(1500)

    pred_closest = list()
    pred_dists = list()

    kdtree, centroids, triangle_indices_list = build_triangle_centroid_kdtree(mesh)

    for idx, p in enumerate(test_pcd):
        closest, dist = closest_point_on_mesh_fast(p, mesh, kdtree, triangle_indices_list, num_neighbors=100)
        pred_closest.append(closest)
        pred_dists.append([dist])

    pred_closest = np.array(pred_closest)
    pred_dists = np.array(pred_dists)
    with open(f"{CUR_DIR}/test_closest_point_results_fast.txt", "w") as file:
        mse = np.average((pred_closest-nearest_points)**2)
        file.write(f"MSE: {mse}\n")
        file.writelines([f"{x} - {y}\n" for x, y in zip(pred_closest, nearest_points)])
    assert np.isclose(np.average((pred_closest-nearest_points)**2) + np.average((pred_dists-distances)**2), 0)

