class KDTreeNode:
    """Node in the KDTree."""

    def __init__(self, point_indices, depth=0):
        self.point_indices = point_indices  # Indices of points in this node
        self.depth = depth
        self.axis = depth % 3  # The axis that data is being split by, % 3 because x, y, z
        self.left, self.right, self.median = None, None, None

    def is_leaf(self):
        """Check if the node is a leaf

        Returns:
            bool: True if KDTreeNode is a leaf, False otherwise
        """

        return self.left is None and self.right is None