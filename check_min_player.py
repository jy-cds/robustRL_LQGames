import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from potential import DP_inv, DP,D2P
from functions import gradient
import statistics

# DATA
# A = np.array([[0.956488, 0.0816012,-0.0005],[0.0741349, 0.94121, -0.000708383],[0,0,0.132655]])
# B = np.array([[-0.00550808], [-0.096], [0.867345]])
# C = np.array([[0.00951892], [0.0038373], [0.001]])


A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
C = np.zeros(shape=(2,1)) # np.array([[0.5],[1]])


nx,nu = B.shape
_,nw = C.shape
T = 15

Ru = np.eye(nu)
Rw = 100*np.eye(nw)
Q = np.eye(nx)

safeguard = 2





np.random.seed(2170)
K = 0.001*np.random.normal(size = (nu,nx))
# K = np.array([[ 0.26122728,  0.41846154, -0.06783688]])


L = np.zeros(shape = (nw,nx))
# L = 0.001*np.random.normal(size = (nu,nx))


############### CHECK FOR VARIANCE OF DX, DXY ###############################
# l=50
# DK_list = np.zeros((nx,1))
# DL_list = np.zeros((nx,1))
# Dxy_list = np.zeros((l,nx,nx))
#
#
# for n in range(l):
#     DK,DL,Dxy = gradient(100,300,A,B,C,Q,Ru,Rw,K,L,T)
#
#     DK = DK.reshape((nx, 1))
#     DL = DL.reshape((nx, 1))
#
#
#     Dxy_list[n,:,:] = Dxy
#     DL_list = np.hstack((DL_list,DL))
#     DK_list = np.hstack((DK_list,DK))
#
#
# print('standard deviation of Dy(1): ' ,  statistics.pstdev(DL_list[0,:]))
# print('standard deviation of Dy(2): ' ,  statistics.pstdev(DL_list[1,:]))
# print('mean of Dy(1)', np.mean(DL_list[0,:]))
# print('mean of Dy(2)', np.mean(DL_list[1,:]))
# print('------------------------------------')
#
# print('standard deviation of Dx(1): ' ,  statistics.pstdev(DK_list[0,:]))
# print('standard deviation of Dx(2): ' ,  statistics.pstdev(DK_list[1,:]))
# print('mean of Dx(1)', np.mean(DK_list[0,:]))
# print('mean of Dx(2)', np.mean(DK_list[1,:]))
# print('------------------------------------')
# print('standard deviation of Dxy(1): ' ,  statistics.pstdev(Dxy_list[:,0,0]))
# print('standard deviation of Dxy(2): ' ,  statistics.pstdev(Dxy_list[:,0,1]))
# # print('standard deviation of Dxy(3): ' ,  statistics.pstdev(Dxy_list[:,0,2]))
# print('mean of Dxy(1)', np.mean(Dxy_list[:,0,0]))
# print('mean of Dxy(2)', np.mean(Dxy_list[:,0,1]))
# # print('mean of Dxy(3)', np.mean(Dxy_list[:,0,2]))
#
#
# p1 = plt.figure(1)
# plt.subplot(131)
# plt.plot(range(l+1),DK_list[0,:])
#
# plt.subplot(132)
# plt.plot(range(l+1),DK_list[1,:])


# plt.subplot(133)
# plt.plot(range(l+1),DK_list[2,:])
# plt.title('Dx')
#
# p2 = plt.figure(2)
# plt.subplot(231)
# plt.plot(range(l),Dxy_list[:,0,0])
#
# plt.subplot(232)
# plt.plot(range(l),Dxy_list[:,0,1])
#
#
# plt.subplot(233)
# plt.plot(range(l),Dxy_list[:,0,2])
#
# plt.subplot(234)
# plt.plot(range(l),Dxy_list[:,1,0])
#
# plt.subplot(235)
# plt.plot(range(l),Dxy_list[:,1,1])
#
# plt.subplot(236)
# plt.plot(range(l),Dxy_list[:,1,2])
#
# plt.title('Dxy')

plt.show()


####################### GRADIENT DESCENT OF K #######################################
#
l = 1000
K_list = np.zeros(shape = (nx,l))
for n in range(l):
    DK,DL,Dxy = gradient(50,200,A,B,C,Q,Ru,Rw,K,L,T)
    K = K - 0.002 * DK
    K = np.minimum(np.maximum(K, -safeguard), safeguard)
    K_list[:,n] = K.flatten()
    if n % 10 == 0:
        print('-----',n,'------')
        print('Dx: ',DK)
        print('K: ',K)
        e,_ = np.linalg.eig(A+B@K)
        print('stability: ', np.max(np.abs(e)))

p1 = plt.figure(1)
plt.subplot(131)
plt.plot(range(l),K_list[0,:])

plt.subplot(132)
plt.plot(range(l),K_list[1,:])

#
# plt.subplot(133)
# plt.plot(range(l),K_list[2,:])
# plt.title('K')

plt.show()





