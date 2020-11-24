
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from potential import DP_inv, DP,D2P
from functions import gradient, proj
import statistics

# DATA
# A = np.array([[0.956488, 0.0816012,-0.0005],[0.0741349, 0.94121, -0.000708383],[0,0,0.132655]])
# B = np.array([[-0.00550808], [-0.096], [0.867345]])
# C = np.array([[0.00951892], [0.0038373], [0.001]])


A = np.array([[1,1],[0,1]])
B = np.zeros(shape=(2,1))    #np.array([[0],[1]])
C = np.array([[0.5],[1]])

nx,nu = B.shape
_,nw = C.shape
T = 10

Ru = np.eye(nu)
Rw = 100*np.eye(nw)
Q = np.eye(nx)


safeguard = 2

np.random.seed(1025)
q = 0.5
L = 0.001*np.random.normal(size = (nu,nx))
# K = np.array([[ 0.26122728,  0.41846154, -0.06783688]])


K = np.zeros(shape = (nw,nx))
# L = 0.001*np.random.normal(size = (nu,nx))

############### GRADIENT DESCENT OF MAX PLAYER #########################

#
l = 500
L_list = np.zeros(shape = (nx,l))
for n in range(l):
    DK,DL,Dxy = gradient(100,200,A,B,C,Q,Ru,Rw,K,L,T)
    L = L - 0.0001 * DL
    L = np.minimum(np.maximum(L, -safeguard), safeguard)
    L_list[:,n] = L.flatten()
    e,_ = np.linalg.eig(Q-(L.T@Rw)@L)
    if n % 1 == 0:
        print('-----',n,'------')
        print('Constraint: ', np.min(e) )
        print('Dy: ',DL)
        print('L: ',L)
        e,_ = np.linalg.eig(A+C@L)
        print('stability: ', np.max(np.abs(e)))

p1 = plt.figure(1)
plt.subplot(131)
plt.plot(range(l),L_list[0,:])

plt.subplot(132)
plt.plot(range(l),L_list[1,:])

#
# plt.subplot(133)
# plt.plot(range(l),K_list[2,:])
# plt.title('K')

plt.show()
