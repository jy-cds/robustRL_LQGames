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
# A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
# B = np.eye(3)
# R = np.identity(3)
# Q = np.eye(3)
# T = 15

# Double Integrator
A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
C = np.array([[0.5],[1]])


seed(1025)
# rng = np.random.default_rng(323)
# temp = loadmat('A.mat')
# A = temp['A']
# nx,_ = A.shape
#
# N = nx
# k = (np.floor(N*(1-0.85))).astype(int)
# arr = np.array([0] * k + [1] * (N-k))
# rng.shuffle(arr)
# B = arr.reshape((nx,1))
#
# k = (np.floor(N*(1-0.65))).astype(int)
# arr = np.array([0] * k + [1] * (N-k))
# rng.shuffle(arr)
# C = arr.reshape((nx,1))
#

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



#initialization
K = 0.001*np.random.normal(size = (nu,nx))
L = 0.001*np.random.normal(size = (nu,nx))
Lambda = 0.01*np.random.rand(nx,nx)
Lambda = np.eye(nx)+ Lambda + Lambda.T
Lambda = proj(Lambda,2)
Lambda = Lambda.reshape((nx**2,1))
dual_l = DP(Lambda, nx) #dual space



# Step size defined
eta_x = 2e-4
eta_y = 5e-3

############################ CGD Main Loop ##################################
N = 6000
# CMD lists

L_list = np.zeros((nx,1))
K_list = np.zeros((nx,1))
lambda_list = np.zeros((len(Lambda),1))
DK_list = np.zeros((nx,1))
DL_list = np.zeros((nx,1))
Dlambda_list = np.zeros((len(Lambda),1))
Dxy_list = np.zeros(shape=(N,nx**2+nx,nx))
constraint_list = []
psd_list = []

for i in range(N):
    # Assign dual variables from primal variables DUAL
    x = np.vstack((K.T, dual_l))
    y = L.T

    # Get gradients and hessians for local NE computation PRIMAL
    DK,DL,DKL = gradient(50,200,A,B,C,Q,Ru,Rw,K,L,T)
    Dlambda = Df_lambda(Lambda,L,Q,q,Rw,nx)
    DfL = Df_L(Lambda,L,Q,q,Rw,nx)
    DflL = Df_lambda_L(Lambda,L,Q,q,Rw,nx).reshape((nx**2,nx))

    # Dx,Dy are all column vectors PRIMAL
    Dx = np.vstack((DK.reshape((nx,1)),Dlambda))
    Dy = DL + DfL
    Dy = Dy.T
    Dxy = np.vstack((DKL,DflL))
    Dyx = Dxy.T
    hessian = LA.block_diag(np.eye(nx), D2P(Lambda,nx).reshape((nx**2,nx**2)))
    hessian_inv = np.linalg.inv(hessian)

    # Local NE PRIMAL
    Jx = np.linalg.inv(1/eta_x * hessian + eta_y * np.matmul(Dxy,Dyx))
    Jy = np.linalg.inv(1/eta_y * np.eye(nx) + eta_x * np.matmul(np.matmul(Dyx,hessian_inv), Dxy))
    del_x = -np.matmul(Jx,(Dx + eta_y * np.matmul(np.matmul(Dxy,np.eye(nx)),Dy)))
    del_y =  np.matmul(Jy,(Dy - eta_x * np.matmul(np.matmul(Dyx,hessian_inv),Dx)))

    # Storing CMD iteration to lists before updating
    Dlambda_list = np.hstack((Dlambda_list,Dlambda))
    lambda_list = np.hstack((lambda_list,Lambda))
    L_list =  np.hstack((L_list,L.reshape((nx,nw))))
    K_list = np.hstack((K_list,K.reshape((nx,nu))))
    DK_list = np.hstack((DK_list,DK.T))
    DL_list =  np.hstack((DL_list,Dy))
    Dxy_list[i,:,:] = Dxy

    # dual space update DUAL
    x = x + hessian @ del_x
    y = y + del_y


    # Reassignment from dual variable to primal variables PRIMAL
    K = x[0:nx,0].T # first nx elements are K
    K = np.minimum(np.maximum(K, -safeguard), safeguard)
    K = K.reshape((nu,nx))
    dual_l = x[nx:,0].reshape((nx,nx))
    dual_l = (dual_l+dual_l.T)/2
    dual_l = dual_l.reshape((nx**2,1))
    Lambda = DP_inv(dual_l,nx) # get primal Lambda from the dual update

    b, _ = np.linalg.eig(Lambda.reshape((nx,nx)))
    psd_list.append(np.min(b))
    L = y.reshape((nw,nx))
    e, _ = np.linalg.eig(Q - L.T @ Rw @ L)
    constraint_list.append(np.min(e))


    if i%1000 ==0:
        print('-------------',i,'-------------------')
        print('K is ',K)
        print('L is ',L)
        print('lambda is ', np.min(b))
        print('Constraint is ', np.min(e))
        print('DK is ', DK)
        print('DL is ', DL)
        # print('Dxy is ', Dxy)

        np.save('L_list', L_list)
        np.save('K_list', K_list)
        np.save('lambda_list', lambda_list)
        np.save('constraint_list', constraint_list)
        np.save('DK_list', DK_list)
        np.save('DL_list', DL_list)
        np.save('Dlambda_list', Dlambda_list)
        np.save('Dxy_list', Dxy_list)
        np.save('psd_list', psd_list)



print('K_CMD is :',K)
print('L_CMD is :',L)
print('lambda is : ', psd_list[-1])


K_list = np.load('K_list.npy')
L_list = np.load('L_list.npy')
DK_list = np.load('DK_list.npy')
DL_list = np.load('DL_list.npy')
Dxy_list = np.load('Dxy_list.npy')



p1 = plt.figure(1)
plt.subplot(121)
plt.plot(K_list[0,:],label = 'CMD')
plt.legend()

plt.subplot(122)
plt.plot(K_list[1,:],label = 'CMD')
plt.legend()
plt.title('K')

p2 = plt.figure(2)
plt.subplot(121)
plt.plot(L_list[0,:], label = 'CMD')
plt.legend()


plt.subplot(122)
plt.plot(L_list[1,:],label = 'CMD')
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


