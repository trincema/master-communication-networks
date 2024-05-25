def zigzag_scan(matrix):
    rows, cols = len(matrix), len(matrix[0])
    result = []
    row, col = 0, 0

    for _ in range(rows * cols):
        result.append(matrix[row][col])

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

    return result

# Example usage:
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(zigzag_scan(matrix))
