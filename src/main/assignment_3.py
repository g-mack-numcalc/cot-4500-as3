def dydt(t, y):
    return t - y**2

def euler_method(dydt, t0, y0, h, n):
    t = t0
    y = y0

    for _ in range(n):
        y += h * dydt(t, y)
        t += h

    return y

t0 = 0
y0 = 1
t_final = 2
n = 10
h = (t_final - t0) / n

y_approx = euler_method(dydt, t0, y0, h, n)

print(y_approx)




def runge_kutta_method(f, t0, y0, h, n):
    def rk4(f, t, y, h):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    t = t0
    y = y0
    for _ in range(n):
        y = rk4(f, t, y, h)
        t += h
    return y


def dydt(t, y):
    return t - y ** 2


t0 = 0
y0 = 1
t_final = 2
n = 10
h = (t_final - t0) / n

result = runge_kutta_method(dydt, t0, y0, h, n)
print(result)




import numpy as np

def gaussian_elimination(matrix):
    n = len(matrix)
    
    for i in range(n):
        max_element = abs(matrix[i][i])
        max_row = i
        
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > max_element:
                max_element = abs(matrix[k][i])
                max_row = k

        matrix[[i, max_row]] = matrix[[max_row, i]]

        for k in range(i + 1, n):
            factor = matrix[k][i] / matrix[i][i]
            matrix[k, i:] -= factor * matrix[i, i:]
    
    return matrix

def backward_substitution(matrix):
    n = len(matrix)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (matrix[i][-1] - np.sum(matrix[i, i+1:n] * x[i+1:n])) / matrix[i][i]
    
    return x

augmented_matrix = np.array([[2, -1, 1, 6],
                             [1, 3, 1, 0],
                             [-1, 5, 4, -3]], dtype=float)

eliminated_matrix = gaussian_elimination(augmented_matrix)
solution = backward_substitution(eliminated_matrix)

print(solution)



import numpy as np

# define the matrix
A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

# define the dimensions of the matrix
m, n = A.shape

# initialize L and U matrices
L = np.eye(n)
U = np.zeros((n, n))

# perform LU factorization
for j in range(n):
    U[0,j] = A[0,j]
    
for i in range(1, n):
    L[i,0] = A[i,0] / U[0,0]
    
for i in range(1, n):
    for j in range(i, n):
        s1 = sum(U[k,j] * L[i,k] for k in range(i))
        U[i,j] = A[i,j] - s1
        
    for j in range(i+1, n):
        s2 = sum(U[k,i] * L[j,k] for k in range(i))
        L[j,i] = (A[j,i] - s2) / U[i,i]

# calculate the determinant
det = np.prod(np.diag(U))

# print out the determinant, L matrix, and U matrix
print(det)

print(L)

print(U)




# Define the matrix
matrix = [
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
]

# Check if the matrix is diagonally dominant
is_diagonally_dominant = True
for i in range(len(matrix)):
    row_sum = sum(abs(matrix[i][j]) for j in range(len(matrix[i])) if j != i)
    if abs(matrix[i][i]) <= row_sum:
        is_diagonally_dominant = False
        break


# Print the result
if is_diagonally_dominant:
    print("True")
else:
    print("False")



import numpy as np

# define the matrix
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

# calculate the eigenvalues of A
eigenvalues = np.linalg.eigvals(A)

# check if all eigenvalues are positive
if all(eigenvalues > 0):
    print("True")
else:
    print("False")
