# Matrix-Class

> A personal linear algebra library inspired by MIT's 18.06 course.

## About

This project is a **Python** implementation of a `Matrix` class that I wrote to apply the concepts I learned from **MIT's 18.06 Linear Algebra** course. It contains several functionalities like **Row Reduced Echelon Form (RREF)**, **matrix multiplication**, **nullspace computation**, **solving systems of linear equations**, and *more*. Built from scratch without using external libraries. 

## Features

- Matrix creation and validation
- Matrix addition, subtraction, scalar multiplication
- Dot product and matrix multiplication
- Transposition
- Row reduction to RREF
- Rank and pivot column computation
- Nullspace (kernel) basis computation
- Solve systems of linear equations
  - Detects unique, infinite, or no solutions
  - Finds the general solution for infinite solutions
- Identity matrix generation
- Matrix augmentation with vectors
- Detects and pushes zero rows to the bottom

## Example Usage

### Infinite solution example
```py
matrix = Matrix([[2, 2, 2], [2, 3, 2], [1, 1, 1]]) 
b = [-2, 4, -1]
matrix.solve_equations(b)

# Output: 

# Infinite solutions. General solution:
# --------------------------------------
# x1: -1.0a + -7.0
# x2: 6.0
# x3: a 
```

### Unique solution example
```py
matrix = Matrix([[7, 5], [3, -4]])
b = [-12, 1]
matrix.solve_equations(b)

# Output:

# Unique solution:
# --------------------------------------
# x1 = -1.0 x2 = -1.0 
```

### No solution example
```py
matrix = Matrix([[2,4], [4,8]])
b = [-1,2]
matrix.solve_equations(b)

# Output:

# No solution.
```

## Disclaimer
This project is for educational purposes only. It is not optimized for large-scale computations.

