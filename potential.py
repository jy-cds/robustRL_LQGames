import autograd.numpy as np
import autograd.scipy.linalg as LA
from autograd import grad, jacobian
from autograd import elementwise_grad as egrad


##################### SINGLE CONSTRAINT CASE #############################
# # NOTE LAMBDA AS THE INPUT IS ALWAYS A VECTOR (Nx**2 by 1)
# # P = logdet(M) + 1/2*norm(M)^2
# def P(Lambda,nx):
#     Lambda = np.reshape(Lambda,(nx,nx))
#     # return np.trace(np.dot(Lambda_stack,LA.logm(Lambda_stack)))
#     sign, logdet = np.linalg.slogdet(Lambda)
#     return -logdet + 0.5*(np.linalg.norm(Lambda,'fro'))**2
#
# DP = grad(P,0) #vector
# def DP_hand(vec,nx):
#     temp = np.reshape(vec,(nx,nx))
#     return (-np.linalg.inv(temp).T + temp).reshape(nx**2,1)
#
# D2P = jacobian(DP,0) #matrix
#
#
# def DP_inv(vec,nx):
#     Y = np.reshape(vec,(nx,nx))
#     U,s,Vh = np.linalg.svd(Y)
#     s_x = np.empty_like(s)
#     for i in range(len(s)):
#         s_x[i] = np.roots([1,-s[i],-1])[np.roots([1,-s[i],-1])>0]
#
#     return  (U @ np.diag(s_x) @ Vh).reshape((nx**2,1))

##################### MIXED CONSTRAINED & UNCONSTRAINED CASE #############################
# P = logdet(M) + 1/2*norm(M)^2
def P(Lambda,nx):
    Lambda = np.reshape(Lambda,(nx,nx))
    # return np.trace(np.dot(Lambda_stack,LA.logm(Lambda_stack)))
    sign, logdet = np.linalg.slogdet(Lambda)
    return -logdet + 0.5*(np.linalg.norm(Lambda,'fro'))**2 #+ 0.5*(np.linalg.norm(K,'fro'))**2

DP = grad(P,0) #vector
def DP_hand(vec,nx):
    temp = np.reshape(vec,(nx,nx))
    return (-np.linalg.inv(temp).T + temp).reshape(nx**2,1)

D2P = jacobian(DP,0) #matrix


def DP_inv(vec,nx):
    Y = np.reshape(vec,(nx,nx))
    # Y = (Y+Y.T)/2
    s,U = np.linalg.eigh(Y)
    s_x = np.empty_like(s)
    for i in range(len(s)):
        s_x[i] = np.roots([1,-s[i],-1])[np.roots([1,-s[i],-1])>0]
    # print(s_x)
    TT = (U @ np.diag(s_x) @ np.linalg.inv(U))
    b, _ = np.linalg.eig(TT.reshape((nx, nx)))
    # print(b)
    return  TT.reshape((nx**2,1))










##################### UNCONSTRAINED L2 #############################
# # Potential Function: composite function that takes in R3 and output R
# def P(x,r):
#     return 0.5*x**2
#
# # Gradient of Potential function P with respect to vector input. Should be 3x1 vector
# DP = grad(P,0)
#
# # Hessian of Potential function P.  3x3 matrix
# D2P = jacobian(DP,0)
#
# #Conjugate Function:
# def DP_inv(a,r):
#     return 1
#
#
