import numpy as np
import scipy.linalg as LA
from numpy.random import seed
import matplotlib.pyplot as plt
import statistics
from functions import gradient, proj, proj_sgd
from potential import DP_inv, DP,D2P
from gradients import Df_lambda,Df_L,Df_lambda_L
from scipy.io import loadmat


# DATA
A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
B = np.eye(3)
R = np.identity(3)
Q = np.eye(3)
T = 15


A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
C = np.array([[0.5],[1]])




seed(1025)
# rng = np.random.default_rng(323)
# temp = loadmat('A.mat')
# A = temp['A']
# nx,_ = A.shape
#
# N = 20
# k = (np.floor(20*(1-0.85))).astype(int)
# arr = np.array([0] * k + [1] * (N-k))
# rng.shuffle(arr)
# B = arr.reshape((nx,1))
#
# k = (np.floor(20*(1-0.6))).astype(int)
# arr = np.array([0] * k + [1] * (N-k))
# rng.shuffle(arr)
# C = arr.reshape((nx,1))


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
L = 0.001*np.random.normal(size = (nu,nx))



# Step size defined
eta_x = 2e-5
eta_y = 2e-5

############################ CGD Main Loop ##################################
N = 10000

# SGD lists and initialization
L_SGD_list = np.zeros((nx,1))
K_SGD_list = np.zeros((nx,1))
constraint_SGD_list = []

K_SGD = K
L_SGD = L

for i in range(N):
    # SGD update and storing
    DK,DL,DKL = gradient(50,200,A,B,C,Q,Ru,Rw,K_SGD,L_SGD,T)

    K_SGD_list = np.hstack((K_SGD_list,K_SGD.reshape((nx,nu))))
    L_SGD_list = np.hstack((L_SGD_list, L_SGD.reshape((nx, nu))))
    p, _ = np.linalg.eig(Q - L_SGD.T @ Rw @ L_SGD)
    constraint_SGD_list.append(p)

    #update
    K_SGD = K_SGD - eta_x * DK
    L_SGD = L_SGD + eta_y * DL
    temp = L_SGD
    L_SGD = proj_sgd(L_SGD.T,l_max).T






    if i%100 ==0:
        print('-------------',i,'-------------------')
        print("projected L ", L_SGD, " with prjection bound ", l_max)
        print('K is ',K_SGD)
        print('L is ',L_SGD)
        print('Constraint is ', np.min(p))

        np.save('constraint_SGD_list',constraint_SGD_list)
        np.save('K_SGD_list', K_SGD_list)
        np.save('L_SGD_list', L_SGD_list)



print('K_SGD is :',K_SGD)
print('L_SGD is :',L_SGD)


K_list = np.load('K_list.npy')
L_list = np.load('L_list.npy')
DK_list = np.load('DK_list.npy')
DL_list = np.load('DL_list.npy')
Dxy_list = np.load('Dxy_list.npy')
K_SGD_list = np.load('K_SGD_list.npy')
L_SGD_list = np.load('L_SGD_list.npy')


p1 = plt.figure(1)
plt.subplot(121)
plt.plot(K_list[0,:],label = 'CMD')
plt.plot(K_SGD_list[0,:],label = 'SGD')
plt.legend()

plt.subplot(122)
plt.plot(K_list[1,:],label = 'CMD')
plt.plot(K_SGD_list[1,:],label = 'SGD')
plt.legend()
plt.title('K')

p2 = plt.figure(2)
plt.subplot(121)
plt.plot(L_list[0,:], label = 'CMD')
plt.plot(L_SGD_list[0,:],label = 'SGD')
plt.legend()


plt.subplot(122)
plt.plot(L_list[1,:],label = 'CMD')
plt.plot(L_SGD_list[1,:],label = 'SGD')
plt.title('L')
plt.legend()


#
# p3 = plt.figure(3)
# plt.subplot(121)
# plt.plot(DK_list[0,2000:])
#
# plt.subplot(122)
# plt.plot(DK_list[1,2000:])
# plt.title('DK')
#
#
# p4 = plt.figure(4)
# plt.subplot(121)
# plt.plot(DL_list[0,2000:])
#
# plt.subplot(122)
# plt.plot(DL_list[1,2000:])
# plt.title('DL')
#
#
# p5 = plt.figure(5)
# plt.plot(constraint_list)
# plt.title('smallest eigenvalue of Q-LL*Rw')

plt.show()


