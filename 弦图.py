import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

def chord_diagram(matrix, labels=None, colors=None, ax=None):
    # If no axis is provided, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')

    # Normalize the matrix
    matrix = np.array(matrix)
    row_sum = matrix.sum(axis=1)
    matrix = matrix / row_sum[:, np.newaxis]

    # Number of nodes
    n = len(matrix)
    PI = np.pi

    # Compute the positions of the arcs
    arc_positions = np.cumsum(row_sum) / row_sum.sum() * 2 * PI
    arc_positions = np.insert(arc_positions, 0, 0)

    # Draw the arcs
    for i in range(n):
        start_angle = arc_positions[i]
        end_angle = arc_positions[i + 1]
        angle = end_angle - start_angle

        # Draw the arc
        arc = patches.Arc((0, 0), 2, 2, theta1=np.degrees(start_angle),
                          theta2=np.degrees(end_angle), color='black')
        ax.add_patch(arc)

        # Draw the text label
        if labels:
            angle_middle = (start_angle + end_angle) / 2
            x = 1.2 * np.cos(angle_middle)
            y = 1.2 * np.sin(angle_middle)
            ax.text(x, y, labels[i], ha='center', va='center')

    # Draw the chords
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] > 0:
                start_angle_i = (arc_positions[i] + arc_positions[i + 1]) / 2
                start_angle_j = (arc_positions[j] + arc_positions[j + 1]) / 2

                path = Path([
                    (np.cos(start_angle_i), np.sin(start_angle_i)),
                    (0, 0),
                    (np.cos(start_angle_j), np.sin(start_angle_j))
                ], [Path.MOVETO, Path.CURVE3, Path.CURVE3])

                patch = patches.PathPatch(path, facecolor='none', edgecolor='gray', lw=matrix[i, j] * 5)
                ax.add_patch(patch)

    plt.show()

# Example usage
matrix = [
    [0, 2, 3, 4],
    [2, 0, 5, 1],
    [3, 5, 0, 2],
    [4, 1, 2, 0]
]
labels = ['A', 'B', 'C', 'D']

chord_diagram(matrix, labels)