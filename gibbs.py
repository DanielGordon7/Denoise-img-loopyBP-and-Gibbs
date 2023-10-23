import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def get_markov_blanket(x_m, row, col) -> list:
    markov_blanket = []
    # corners
    if row == 0 and col == 0:
        markov_blanket = [x_m[row+1,col], x_m[row,col+1]]
    elif row == 0 and col == n-1:
        markov_blanket = [x_m[row+1,col], x_m[row,col-1]]
    elif row == m-1 and col == 0:
        markov_blanket = [x_m[row-1,col], x_m[row,col+1]]
    elif row == m-1 and col == n-1:
        markov_blanket = [x_m[row-1,col], x_m[row,col-1]]
    # edges
    elif row == 0 and col not in [0,n-1]:
        markov_blanket = [x_m[row+1,col], x_m[row,col-1], x_m[row,col+1]]
    elif row == m-1 and col not in [0,n-1]:
        markov_blanket = [x_m[row-1,col], x_m[row,col-1], x_m[row,col+1]]
    elif col == 0 and row not in [0,m-1]:
        markov_blanket = [x_m[row-1,col], x_m[row+1,col], x_m[row,col+1]]
    elif col == n-1 and row not in [0,m-1]:
        markov_blanket = [x_m[row-1,col], x_m[row+1,col], x_m[row,col-1]]
    else:
        markov_blanket = [x_m[row-1,col], x_m[row+1,col], x_m[row,col-1], x_m[row,col+1]]
    return markov_blanket

def phi(x, y):
    mu = mu_m[x]
    sigma_sqr = sigma_sqr_m[x]
    return -1*((y-mu)**2) / (2*sigma_sqr)

def psi(xi, xj):
    beta = 20
    return -1 * beta * ((xi-xj)**2)

def get_gibbs_posterior(xi, y, markov_blanket):
    posterior = phi(xi, y)
    for xj in markov_blanket:
        posterior += psi(xi, xj)
    return posterior  # log

mat_data = sio.loadmat("img.mat")
data = mat_data["img1"]

mu_m = [147, 150]
sigma_sqr_m = [0.5, 0.5]
m = 50
n = 60

np.random.seed(472023)
initial_samples = np.random.binomial(n=1, p=0.5, size=m*n).reshape((m,  n))

y_m = data.copy()
max_num_samples = 200
actual_num_samples = 0
gibbs_samples = np.zeros((m, n, max_num_samples))
gibbs_samples[:, :, 0] = initial_samples.copy()
convergence = 10  # check last 10 samples
burn_in = 100
for itr in range(max_num_samples-1):
    cur_sample = gibbs_samples[:, :, itr]
    for row in range(m):
        for col in range(n):
            log_probs = []
            markov_blanket = get_markov_blanket(cur_sample, row, col)
            y = y_m[row,col]
            for xi in [0, 1]:
                log_probs.append(get_gibbs_posterior(xi, y, markov_blanket))

            gibbs_posterior = np.exp(log_probs[1]) / (np.exp(log_probs[0]) + np.exp(log_probs[1])) 
            sample = np.random.binomial(n=1, p=gibbs_posterior)
            gibbs_samples[row, col, itr+1] = sample
    actual_num_samples += 1
    if itr > burn_in and np.sum(np.diff(gibbs_samples[5, 5, itr-convergence:itr])) ==  0 and \
        np.sum(np.diff(gibbs_samples[m-5, n-5, itr-convergence:itr])) ==  0 and np.sum(np.diff(gibbs_samples[25, 30, itr-convergence:itr])) ==  0:
        break

imgplot = plt.imshow(gibbs_samples[:, :, actual_num_samples],cmap='gray')
plt.show()