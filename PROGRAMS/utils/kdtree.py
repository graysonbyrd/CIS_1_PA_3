import numpy as np

from .kdtree_node import KDTreeNode

class KDTree:
    """Custom KDTree Implementation"""

    def __init__(self, points: np.ndarray):
        self.points = points  # Shape (N, 3) or will not work
        self.root = self.build_kdtree(np.arange(len(points), dtype=int))

    def build_kdtree(self, point_indices, depth=0):
        if len(point_indices) == 0:
            return None
        axis = depth % 3 # represents x, y, z, axis is used to split data through each dimension

        # Sort point indices along the current axis, if depth=0, sorts by x-axis, if depth=1, sorts by y-axis
        # if depth=2, sorts by z-axis
        sorted_indices = point_indices[np.argsort(self.points[point_indices, axis])]

        # Find the median index
        median_idx = len(sorted_indices) // 2
        median_point_idx = sorted_indices[median_idx]

        # Create node with the median point index
        node = KDTreeNode(point_indices=[median_point_idx], depth=depth)
        node.axis = axis
        node.median = self.points[median_point_idx, axis]

        # Recursively build left and right subtrees
        node.left = self.build_kdtree(sorted_indices[:median_idx], depth + 1) # nodes with axis value less than median, depth + 1
        node.right = self.build_kdtree(sorted_indices[(median_idx + 1):], depth + 1) # nodes with axis value greater than median, depth + 1
        return node

    def recursive_search(self, node: KDTreeNode, point: np.ndarray, best: list, k: int):
        if node is None:
            return
        axis = node.axis
        # Decide which side to search first
        if point[axis] < node.median:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        # Search down the tree
        self.recursive_search(first, point, best, k)

        # Check current node's median point
        idx = node.point_indices[0]
        dist = np.linalg.norm(self.points[idx] - point)
        if len(best) < k:
            best.append((dist, int(idx)))
            best.sort(key=lambda x: x[0])
        elif dist < best[-1][0]:
            best[-1] = (dist, int(idx))
            best.sort(key=lambda x: x[0])

        # Check if we need to search the other side
        if len(best) < k or abs(point[axis] - node.median) < best[-1][0]:
            self.recursive_search(second, point, best, k)

    def query(self, point: np.ndarray, k=1):
        best = []
        self.recursive_search(self.root, point, best, k)

        # Sort best by dist
        best.sort(key=lambda x: x[0])

        # need to return distances and indices
        indices = []
        distances = []
        for dist, idx in best:
            indices.append(idx)
            distances.append(dist)
        return distances, indices
