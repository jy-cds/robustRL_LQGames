from potential import P,DP, DP_hand, D2P, DP_inv
import numpy as np
np.random.seed(2)


vec = np.random.randint(4,size=(2,2))
vec = vec.T+vec + 6*np.eye(2) #symmetrize

vec = vec.reshape((4,1))
nx = 2

print(P(vec,nx) )# Checked
print(DP(vec,nx)) # checked
print(DP_hand(vec,nx)) # same as autograd
print(D2P(vec,nx).reshape(nx**2,nx**2))

X = DP_inv(vec,nx)
print(X)
print(vec)
print(DP(X,nx))

