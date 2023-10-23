import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def get_markov_blanket(x_m, row, col) -> list:
    markov_blanket = []  # clockwise order: north, east, south, west
    indices = [(row-1,col), (row,col+1), (row+1,col), (row,col-1)]
    loc = {}
    # corners
    if row == 0 and col == 0:
        loc = dict(zip(indices[1:3], [1,2]))
        markov_blanket = [x_m[indices[1]], x_m[indices[2]]]  # right, down
    elif row == 0 and col == n-1:
        loc = dict(zip(indices[2:4], [2,3]))
        markov_blanket = [x_m[indices[2]], x_m[indices[3]]]  # down, left
    elif row == m-1 and col == 0:
        loc = dict(zip(indices[0:2], [0,1]))
        markov_blanket = [x_m[indices[0]], x_m[indices[1]]]  # up, right
    elif row == m-1 and col == n-1:
        loc = dict(zip([indices[0], indices[3]], [0,3]))
        markov_blanket = [x_m[indices[0]], x_m[indices[3]]]  # up, left
    # edges
    elif row == 0 and col not in [0,n-1]:
        loc = dict(zip(indices[1:4], [1,2,3]))
        markov_blanket = [x_m[indices[1]], x_m[indices[2]], x_m[indices[3]]]  # right, down, left
    elif row == m-1 and col not in [0,n-1]:
        loc = dict(zip([indices[0], indices[1], indices[3]], [0,1,3]))
        markov_blanket = [x_m[indices[0]], x_m[indices[1]], x_m[indices[3]]]  # up, right, left
    elif col == 0 and row not in [0,m-1]:
        loc = dict(zip(indices[0:3], [0,1,2]))
        markov_blanket = [x_m[indices[0]], x_m[indices[1]], x_m[indices[2]]]  # up, right, down
    elif col == n-1 and row not in [0,m-1]:
        loc = dict(zip([indices[0], indices[2], indices[3]], [0,2,3]))
        markov_blanket = [x_m[indices[0]], x_m[indices[2]], x_m[indices[3]]]  # up, down, left
    else:
        loc = dict(zip(indices, [0,1,2,3]))
        markov_blanket = [x_m[indices[0]], x_m[indices[1]], x_m[indices[2]], x_m[indices[3]]]  # up, right, down, left

    return markov_blanket, loc

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
    return posterior

def get_outgoing_msg(neighbors_loc, cur_loc, sending_msgs, msg_recipient_loc, y_m, itr) -> tuple:  # [m(xi->xj=0), m(xi->xj=1)]
    node_potential = 0
    edge_potential = 0
    outgoing_msg = []
    msg_xi = []
    for xj in [0, 1]:
        for xi in [0, 1]:
            node_potential = phi(xi, y_m[cur_loc])
            incoming_msgs = 0
            for nbor_loc in neighbors_loc:
                if nbor_loc == msg_recipient_loc:
                    edge_potential = psi(xi, xj)
                else: 
                    nbor_msg_idx = neighbors_loc[nbor_loc] % 2 if neighbors_loc[nbor_loc] // 2 == 1 else neighbors_loc[nbor_loc] + 2
                    incoming_msgs += sending_msgs[nbor_loc][itr-1][nbor_msg_idx][xi]
                    
            msg_xi.append(np.exp(incoming_msgs + edge_potential + node_potential))
        outgoing_msg.append(np.log(sum(msg_xi)))

    return tuple(outgoing_msg)

def get_updated_belief(neighbors_loc, cur_loc, sending_msgs, y_m, itr) -> tuple:  # [b(xi=0), b(xi=1)]
    all_incoming_msgs = 0
    updated_belief = []
    for xi in [0, 1]:
        node_potential = phi(xi, y_m[cur_loc])
        for nbor_loc in neighbors_loc:
            nbor_msg_idx = neighbors_loc[nbor_loc] % 2 if neighbors_loc[nbor_loc] // 2 == 1 else neighbors_loc[nbor_loc] + 2
            all_incoming_msgs += sending_msgs[nbor_loc][itr-1][nbor_msg_idx][xi]
        updated_belief.append(all_incoming_msgs + node_potential)

    return tuple(updated_belief)

def log_to_prob(distr: tuple) -> tuple:
    normalized_p1 = np.exp(distr[0]) / (np.exp(distr[0]) + np.exp(distr[1]))
    normalized_p2 = 1 - normalized_p1
    return normalized_p1, normalized_p2

def normalize_log(distr: tuple) -> tuple:
    normalized_distr = []
    norm_p1, norm_p2 = log_to_prob(distr)
    normalized_distr.append(np.log(norm_p1))
    normalized_distr.append(np.log(norm_p2))
    return tuple(normalized_distr)



mat_data = sio.loadmat("img.mat")
data = mat_data["img1"]

mu_m = [147, 150]
sigma_sqr_m = [0.5, 0.5]
m = 50
n = 60

beliefs = {}  # key=loc, value=log belief
for i in range(m):
    for j in range(n):
        beliefs[(i,j)] = {0: (0,0)} 
sending_msgs = {}  # key = { matrix index (i,j), value = dict: {key=itr, value=[outgoing msgs]} }
for i in range(m):
    for j in range(n):
        sending_msgs[(i,j)] = {0: [(0,0), (0,0), (0,0), (0,0)]} 


x_m = np.random.binomial(n=1, p=0.5, size=(m, n))  # initialize beliefs randomly
y_m = data.copy()
max_itr = 20
for itr in range(1, max_itr):
    for i in range(m):
        for j in range(n): 
            all_possible_neighbors_loc = [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]
            cur_loc = (i,j)
            neighbors_loc = get_markov_blanket(x_m, i, j)[1]
            sending_msgs[cur_loc][itr] = []

            for msg_recipient_loc in all_possible_neighbors_loc:
                
                if msg_recipient_loc in neighbors_loc:
                    outgoing_msg = get_outgoing_msg(neighbors_loc, cur_loc, sending_msgs, msg_recipient_loc, y_m, itr)
                    sending_msgs[cur_loc][itr].append(normalize_log(outgoing_msg))
                else:
                    sending_msgs[cur_loc][itr].append((0,0))
                
            updated_belief = get_updated_belief(neighbors_loc, cur_loc, sending_msgs, y_m, itr)
            beliefs[cur_loc][itr] = normalize_log(updated_belief)

# sample from normalized beliefs
np.random.seed(1)
samples = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        prob = log_to_prob(beliefs[(i,j)][max_itr-1])[1]
        samples[i,j] = np.random.binomial(n=1, p=prob)

# plot
imgplot = plt.imshow(samples,cmap='gray')
plt.show()