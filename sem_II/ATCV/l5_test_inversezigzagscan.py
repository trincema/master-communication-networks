import numpy as np

def zigzag_scan(matrix):
    rows, cols = matrix.shape
    result = []
    row, col = 0, 0

    for _ in range(rows * cols):
        result.append(matrix[row, col])

        if (row + col) % 2 == 0:  # Even sum of indices
            if col == cols - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:  # Odd sum of indices
            if row == rows - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1

    return np.array(result)

def reverse_zigzag_scan(array, rows, cols):
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    matrix = np.zeros((rows, cols), dtype=array.dtype)
    row, col = 0, 0

    for i in range(rows * cols):
        matrix[row, col] = array[i]

        if (row + col) % 2 == 0:  # Even sum of indices
            if col == cols - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:  # Odd sum of indices
            if row == rows - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1

    return matrix

# Example usage:
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

zigzagged_array = zigzag_scan(matrix)
print("Zigzagged array:", zigzagged_array)

reconstructed_matrix = reverse_zigzag_scan(zigzagged_array, 3, 3)
print("Reconstructed matrix:")
print(reconstructed_matrix)
