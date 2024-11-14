import numpy as np
from typing import Dict, Tuple, Iterable
from .kdtree import KDTree

def find_closest_point(p: np.ndarray, v: np.ndarray, triangles: Iterable[Tuple[int, int, int]]) -> Tuple[np.ndarray, float]:
    min_dist = float('inf')
    closest_point = None
    for triangle_indices in triangles:
        a, b, c = v[triangle_indices[0]], v[triangle_indices[1]], v[triangle_indices[2]]
        cp = closest_point_on_triangle(p, a, b, c)
        distance = np.linalg.norm(p - cp)
        if distance < min_dist:
            min_dist = distance
            closest_point = cp
    return closest_point, min_dist

def closest_point_on_mesh_slow(p: np.ndarray, mesh: Dict) -> Tuple[np.ndarray, float]:
    """Slow version: Find the closest point on the mesh to point p by checking all triangles."""
    return find_closest_point(p, mesh['V'], mesh['i'])

def closest_point_on_mesh_fast(p: np.ndarray, mesh: Dict, kdtree, triangle_indices_list, num_neighbors=5) -> Tuple[np.ndarray, float]:
    """
    Fast version: Use KDTree to find the closest point on the mesh to point p.
    Represents option #3 on slide #10 / page #5 in https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:finding_point-pairs.pdf
    KDTree is a "Hierarchical Data Structure" which works very well for this problem.
    Option 4 (Rotate each level of the tree to align with data) isn't implemented.
    """
    v = mesh['V']
    distances, indices = kdtree.query(p, k=num_neighbors)
    indices = [indices] if num_neighbors == 1 else np.atleast_1d(indices)
    triangles = [triangle_indices_list[idx] for idx in indices]
    return find_closest_point(p, v, triangles)

def build_triangle_centroid_kdtree(mesh: Dict):
    """Build a KDTree of triangle centroids."""
    v = mesh['V']
    i = mesh['i']
    centroids = []
    triangle_indices_list = []
    for idx, triangle_indices in enumerate(i):
        a, b, c = v[triangle_indices[0]], v[triangle_indices[1]], v[triangle_indices[2]]
        centroids.append((a + b + c) / 3.0)
        triangle_indices_list.append(triangle_indices)
    centroids = np.array(centroids)
    kdtree = KDTree(centroids)
    return kdtree, centroids, triangle_indices_list

# Check out page 6 (slide 11 and beyond) of https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:finding_point-pairs.pdf for guidance
def closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Find the closest point on triangle abc to point p."""

    e = 1e-8

    # checks if the point p is one of the vertices a, b, or c
    if np.allclose(p, a, atol=e):
        return a
    if np.allclose(p, b, atol=e):
        return b
    if np.allclose(p, c, atol=e):
        return c

    ab = b - a
    ac = c - a
    ap = p - a

    # checks if a is the closest point
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= e and d2 <= e:
        return a

    # checks if b is the closest point
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    # checks if p is on the edge ab
    vc = d1 * d4 - d3 * d2
    if vc <= e and d1 >= e and d3 <= e:
        v = d1 / (d1 - d3)
        return a + v * ab

    # checks if c is the closest point
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= e and d5 <= d6:
        return c

    # checks if p is on the edge ac
    vb = d5 * d2 - d1 * d6
    if vb <= e and d2 >= e and d6 <= e:
        w = d2 / (d2 - d6)
        return a + w * ac

    # checks if p is on the edge bc
    va = d3 * d6 - d5 * d4
    if va <= e and (d4 - d3) >= e and (d5 - d6) >= e:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    # if none of the above, projects point p onto the face of the triangle
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

