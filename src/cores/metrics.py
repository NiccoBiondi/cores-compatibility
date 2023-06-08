import numpy as np 


def average_compatibility(matrix=None):
    steps = matrix.shape[0]
    position = np.zeros_like(matrix, dtype=bool)
    for j in range(matrix.shape[0]):
        for i in range(j + 1, matrix.shape[1]):
            if matrix[i][j] < matrix[j][j]:
                position[i, j] = True
    max_ac = (steps * (steps-1)) / 2
    if max_ac < 1:
        max_ac = 1
    ac = max_ac - np.sum(position)
    return (1/max_ac) * ac


def average_multimodel_accuracy(matrix):
    return np.mean(matrix[matrix > 0])