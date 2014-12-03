import numpy as np

m = 23
n = 12

x = 5

sim_matrix = np.random.randint(100, size=(m,n))
print sim_matrix

grids = [[None]*x for i in xrange(x)]

for i in xrange(x):
    for j in xrange(x):
        grids[i][j] = [sim_matrix[i*(int(m/x)):(i+1)*(int(m/x)), j*int(n/x):(j+1)*int(n/x)]]

# Add extra rows
n_extra_rows = m % x
rounded_rows = x * int(m/x)

for i in xrange(rounded_rows, rounded_rows + n_extra_rows):
    for j in xrange(x):
        print x - n_extra_rows + i - rounded_rows, j
        grids[x - n_extra_rows + i - rounded_rows][j].append(sim_matrix[i, j*int(n/x):(j+1)*int(n/x)])

# Add extra columns
n_extra_cols = n % x
rounded_cols = x * (int(n/x))

for i in xrange(x):
    for j in xrange(rounded_cols, rounded_cols + n_extra_cols):
        grids[i][x - n_extra_cols + j - rounded_cols].append(sim_matrix[i*int(m/x):(i+1)*int(m/x), j])

dp_sim_matrix = np.zeros((x,x))

def get_min(matrix_list):
    min = float('inf')
    for m in matrix_list:
        if np.min(m) < min:
            min = np.min(m)
    return min

for i in xrange(x):
    for j in xrange(x):
        dp_sim_matrix[i,j] = get_min(grids[i][j])

print dp_sim_matrix
