import numpy as np

#X is the final dimension of the reduced matrix, pass 15 by default but ideally should be varied to see impact
def doDynamicPooling(sim_matrix,x):
    m = sim_matrix.shape[0]
    n = sim_matrix.shape[1]

    if x > m:
        extra = sim_matrix[np.random.choice(range(m),size=x-m,replace=True),:]
        #print extra.shape
        sim_matrix = np.vstack((sim_matrix,extra))
        m = x
        #print sim_matrix.shape

    if x > n:
        extra = sim_matrix[:,np.random.choice(range(n),size=x-n,replace=True)]
        #print extra.shape
        sim_matrix = np.hstack((sim_matrix,extra))
        n = x
        #print sim_matrix.shape

    #x = 5
    grids = [[None]*x for i in xrange(x)]

    for i in xrange(x):
        for j in xrange(x):
            grids[i][j] = [sim_matrix[i*(int(m/x)):(i+1)*(int(m/x)), j*int(n/x):(j+1)*int(n/x)]]

    # Add extra rows
    n_extra_rows = m % x
    rounded_rows = x * int(m/x)

    for i in xrange(rounded_rows, rounded_rows + n_extra_rows):
        for j in xrange(x):
            #print x - n_extra_rows + i - rounded_rows, j
            grids[x - n_extra_rows + i - rounded_rows][j].append(sim_matrix[i, j*int(n/x):(j+1)*int(n/x)])

    # Add extra columns
    n_extra_cols = n % x
    rounded_cols = x * (int(n/x))

    for i in xrange(x):
        for j in xrange(rounded_cols, rounded_cols + n_extra_cols):
            grids[i][x - n_extra_cols + j - rounded_cols].append(sim_matrix[i*int(m/x):(i+1)*int(m/x), j])

    dp_sim_matrix = np.zeros((x,x))


    for i in xrange(x):
        for j in xrange(x):
            dp_sim_matrix[i,j] = get_min(grids[i][j])

    return dp_sim_matrix

def get_min(matrix_list):
    min = float('inf')
    for m in matrix_list:
        if np.min(m) < min:
            min = np.min(m)
    return min



# sim_matrix = np.random.randint(100, size=(23,12))
# print doDynamicPooling(sim_matrix,25)
