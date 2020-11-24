import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from potential import DP_inv, DP,D2P
from functions import gradient
import statistics

A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
C = np.zeros(shape=(2,1))


nx,nu = B.shape
_,nw = C.shape

Ru = np.eye(nu)
Rw = np.eye(nw)
Q = np.eye(nx)
T = 20

safeguard = 2

np.random.seed(2170)
K = 0.001*np.random.normal(size = (nu,nx))
L = np.zeros(shape = (nw,nx))

e,_ = np.linalg.eig(A+B@K)

print('stability: ', np.abs(e))
l = 50
Dx_list = np.zeros((2,1))
Dxy_list = np.zeros((l,nx,nx))

for n in range(l):

    Dx,DL, Dxy = gradient(50,200,A,B,C,Q,Ru,Rw,K,L,T)
    Dx = Dx.reshape((2, 1))

    Dx_list = np.hstack((Dx_list,Dx))
    Dxy_list[n,:,:] = Dxy

print('standard deviation of Dxy(1): ' ,  statistics.pstdev(Dx_list[0,:]))
print('standard deviation of Dxy(2): ' ,  statistics.pstdev(Dx_list[1,:]))
print('mean of Dxy(1)', np.mean(Dx_list[0,:]))
print('mean of Dxy(2)', np.mean(Dx_list[1,:]))

p4 = plt.figure(1)
plt.subplot(121)
plt.scatter(range(l+1),Dx_list[0,:])

plt.subplot(122)
plt.scatter(range(l+1),Dx_list[1,:])
plt.title('Dx')

#
# p5 = plt.figure(2)
# plt.subplot(121)
# plt.scatter(range(31),Dxy_list[0,:])
#
# plt.subplot(122)
# plt.scatter(range(31),Dxy_list[1,:])
# plt.title('Dxy')

plt.show()