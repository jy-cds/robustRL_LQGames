import autograd.numpy as np
from autograd import grad,jacobian
from autograd import elementwise_grad as egrad

# Input L as a ROW VECTOR
def f(Lambda,L,Q,q,Rw,nx):
    Lambda = Lambda.reshape((nx,nx))
    return -np.trace(np.matmul(Lambda,(Rw*np.matmul(L.T,L)-Q-q*np.eye(nx))))

Df_lambda = egrad(f,0)
Df_L = egrad(f,1)
Df_lambda_L = jacobian(Df_lambda,1)
Df_L_lambda = jacobian(Df_L,0)



######################### Testing for Gradients ##################
# np.random.seed(1025)
# Q = np.eye(2)
# L = np.random.rand(1,2)
# Lambda = np.random.rand(2,2)
# Lambda = Lambda + Lambda.T
# Lambda = Lambda.reshape((4,1))
# # print(Lambda)
# Rw = 20
# nx = 2
# q = 0
# print(f(Lambda,L.T,Q,q,Rw,nx))
############ Check D_lambda #############
## 4 by 1 vector
# print(-(Rw*np.matmul(L.T,L)-Q+q*np.eye(nx)).T)
# print(Df_lambda(Lambda,L,Q,q,Rw,nx).reshape((2,2)))
########### Check D_L  ##############
## 1 by 2 vector
# print(Df_L(Lambda,L,Q,q,Rw,nx))
# print(-Rw*(L@(Lambda.reshape((2,2))+Lambda.reshape((2,2)).T)))
#########################
# print(Df_lambda_L(Lambda,L,Q,q,Rw,nx).reshape((4,2)))
# print(Df_L_lambda(Lambda,L,Q,q,Rw,nx).reshape((2,4)))
# print(-2*Rw*L[0])
# print(-Rw*L[0])
