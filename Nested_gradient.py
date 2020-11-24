import numpy as np
import scipy.linalg as LA
from numpy.random import seed
import matplotlib.pyplot as plt
import statistics
from functions import gradient, proj, proj_sgd
from potential import DP_inv, DP,D2P
from gradients import Df_lambda,Df_L,Df_lambda_L

# DATA
# A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
# B = np.eye(3)
# R = np.identity(3)
# Q = np.eye(3)
# T = 15
A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
C = np.array([[0.5],[1]])
nx,nu = B.shape
_,nw = C.shape
T = 15
Ru = np.eye(nu)
Rw = 20*np.eye(nw)
Q = np.eye(nx)
q = 0.01
e,_ = np.linalg.eig(Q)
l_max = (np.min(e) - q) / Rw
# L = np.eye(2)
# L_p = proj(L,l_max)

safeguard = 2
seed(1025)


#initialization
K = 0.001*np.random.normal(size = (nu,nx))
L = np.random.normal(size = (nu,nx))

# K = np.array([[-0.5, -1.4]])
# L = np.array([[0.1,0.2]])

# Step size defined
eta_x = 1e-5
eta_y = 1e-5

############################  Main Loop ##################################
N = 7000

# NGD lists and initialization
L_NGD_list = np.zeros((nx,1))
K_NGD_list = np.zeros((nx,1))
constraint_NGD_list = []
K_NGD = K
L_NGD = L

for i in range(N):
    # NGD update and storing
    K_NGD_list = np.hstack((K_NGD_list, K_NGD.reshape((nx, nu))))
    L_NGD_list = np.hstack((L_NGD_list, L_NGD.reshape((nx, nu))))
    for j in range(100):
        DK, _, _ = gradient(50, 200, A, B, C, Q, Ru, Rw, K_NGD, L_NGD, T)
        # save NGD lists
        p, _ = np.linalg.eig(Q - L_NGD.T @ Rw @ L_NGD)
        constraint_NGD_list.append(p)
        #update
        K_NGD = K_NGD - eta_x * DK

    _, DL, _ = gradient(50, 200, A, B, C, Q, Ru, Rw, K_NGD, L_NGD, T)
    L_NGD = L_NGD + eta_y * DL
    L_NGD = proj_sgd(L_NGD.T,l_max).T


    if i%50 ==0:
        print('-------------',i,'-------------------')
        print('K is ',K_NGD)
        print('L is ',L_NGD)
        print('Constraint is ', np.min(p))

        np.save('K_NGD_list', K_NGD_list)
        np.save('L_NGD_list', L_NGD_list)
        np.save('constraint_NGD_list', constraint_NGD_list)

