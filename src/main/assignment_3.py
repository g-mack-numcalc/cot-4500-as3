import numpy as np

# Function that returns the differential equation t - y^2
def dydt(t, y):
    return t - y**2

# Function that uses Euler's method to numerically approximate the solution of a differential equation
def euler_method(dydt, t0, y0, h, n):
    t = t0
    y = y0

    # Iterate n times to approximate the value of y at the final time
    for _ in range(n):
        y += h * dydt(t, y)
        t += h

    return y

# Function that uses the fourth-order Runge-Kutta method to numerically approximate the solution of a differential equation
def runge_kutta_method(f, t0, y0, h, n):
    def rk4(f, t, y, h):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    t = t0
    y = y0

    # Iterate n times to approximate the value of y at the final time
    for _ in range(n):
        y = rk4(f, t, y, h)
        t += h

    return y

# Function that performs Gaussian elimination on a matrix
def gaussian_elimination(matrix):
    n = len(matrix)

    # Iterate over the rows of the matrix
    for i in range(n):
        # Find the row with the maximum element in the ith column
        max_element = abs(matrix[i][i])
        max_row = i

        for k in range(i + 1, n):
            if abs(matrix[k][i]) > max_element:
                max_element = abs(matrix[k][i])
                max_row = k

        # Swap the ith row with the row with the maximum element in the ith column
        matrix[[i, max_row]] = matrix[[max_row, i]]

        # Use row operations to transform the matrix into row echelon form
        for k in range(i + 1, n):
            factor = matrix[k][i] / matrix[i][i]
            matrix[k, i:] -= factor * matrix[i, i:]

    return matrix

# Function that performs backward substitution on a matrix in row echelon form to solve a system of linear equations
def backward_substitution(matrix):
    n = len(matrix)
    x = np.zeros(n)

    # Iterate over the rows of the matrix in reverse order
    for i in range(n - 1, -1, -1):
        # Use the values of x from the rows below to solve for x_i
        x[i] = (matrix[i][-1] - np.sum(matrix[i, i+1:n] * x[i+1:n])) / matrix[i][i]
    
    return x

# Function that performs LU factorization on a matrix
def lu_factorization(A):
    m, n = A.shape
    L = np.eye(n)
    U = np.zeros((n, n))

    # Compute the entries of U and L
    for j in range(n):
        U[0, j] = A[0, j]

    for i in range(1, n):
        L[i, 0] = A[i, 0] / U[0, 0]

    for i in range(1, n):
        for j in range(i, n):
            s1 = sum(U[k, j] * L[i, k] for k in range(i))
            U[i, j] = A[i, j] - s1

        for j in range(i + 1, n):
            s2 = sum(U[k, i] * L[j, k] for k in range(i))
            L[j, i] = (A[j, i] - s2) / U[i, i]

    return L, U

# Function that checks if a matrix is diagonally dominant
def is_matrix_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        # Compute the sum of the absolute values of the entries in the ith row, except for the diagonal element
        row_sum = sum(abs(matrix[i][j]) for j in range(len(matrix[i])) if j != i)

        # Check if the diagonal element is greater than the sum of the absolute values of the other entries in the row
        if abs(matrix[i][i]) <= row_sum:
            return False

    return True

# Function that checks if a matrix is positive definite
def is_positive_definite(matrix):
    # Compute the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(matrix)

    # Check if all eigenvalues are positive
    return all(eigenvalues > 0)

#1

# Define initial conditions
t0 = 0       # initial time
y0 = 1       # initial value of y

# Define final time and number of steps
t_final = 2  # final time
n = 10       # number of steps

# Calculate step size
h = (t_final - t0) / n   # step size

# Use Euler method to approximate solution of differential equation
y_approx = euler_method(dydt, t0, y0, h, n)

# Print the result to the console
print("%.5f" % y_approx)  # print the approximate value of y to 5 decimal places
print()                   # print an empty line

#2

# Define initial conditions
t0 = 0       # initial time
y0 = 1       # initial value of y

# Define final time and number of steps
t_final = 2  # final time
n = 10       # number of steps

# Calculate step size
h = (t_final - t0) / n   # step size

# Use Runge-Kutta method to approximate solution of differential equation
result = runge_kutta_method(dydt, t0, y0, h, n)

# Print the result to the console
print("%.5f" % result)    # print the approximate value of y to 5 decimal places
print()                   # print an empty line

#3

# Define an augmented matrix
augmented_matrix = np.array([[2, -1, 1, 6],
                             [1, 3, 1, 0],
                             [-1, 5, 4, -3]], dtype=float)

# Use Gaussian elimination to solve the system of linear equations
eliminated_matrix = gaussian_elimination(augmented_matrix)
solution = backward_substitution(eliminated_matrix)

# Print the solution to the console
print(solution)  # print the solution of the system of linear equations
print()           # print an empty line

#4

# Define a matrix
A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

# Use LU factorization to find determinant, lower triangular matrix, and upper triangular matrix
L, U = lu_factorization(A)
det = np.prod(np.diag(U))

# Print the determinant, L, and U to the console
print("%.5f" % det)    # print the determinant of A to 5 decimal places
print()                # print an empty line
print(L)               # print the lower triangular matrix L
print()                # print an empty line
print(U)               # print the upper triangular matrix U
print()                # print an empty line

#5

matrix = [
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
]

# Check if the matrix is diagonally dominant
is_diagonally_dominant = is_matrix_diagonally_dominant(matrix)

# Print the result to the console
if is_diagonally_dominant:
    print("True")    # print "True" if the matrix is diagonally dominant
else:
    print("False")   # print "False" otherwise

print()              # print an empty line

#6

A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

# Check if the matrix is positive definite
is_pd = is_positive_definite(A)

# Print the result to the console
if is_pd:
    print("True")    # print "True" if the matrix is positive definite
else:
    print("False")   # print "False" otherwise