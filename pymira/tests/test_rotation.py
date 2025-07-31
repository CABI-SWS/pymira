# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 07:56:44 2017

@author: simon
"""

import numpy as np
mat = np.asmatrix
norm = np.linalg.norm

from network_stl import *

#A = mat([1,0,0])
#B = mat([0,0.8,0.2])
A = mat([1, 2, 0])
B = mat([1, -2, 0])
A = A/np.linalg.norm(A)
B = B/np.linalg.norm(B)
print('A={}'.format(A))
print('B={}'.format(B))

#0.9839 0.1789 0
# -0.1789 0.9839 0
# 0   0     1.0000

#R = vector_rotation(A, B)
#Ap = np.dot(A,R.T)
#Bp = np.dot(B,R.T)

U = align_vector_rotation(A,B)

#G = np.asmatrix([[A*B.T, mat(-np.linalg.norm(np.cross(A,B))), 0.],
#     [np.linalg.norm(np.cross(A,B)), A*B.T,  0.],
#     [0., 0., 1.],
#     ])
#Fi = np.asarray([ A , (B - (A*B.T)*A) / norm((B - (A*B.T)*A)) , np.cross(B,A) ])
#Fi = np.matrix(np.squeeze(Fi))
#U = Fi * G * np.linalg.inv(Fi)

print U
Ap = np.transpose(U*A.T)
print np.linalg.norm(B-np.transpose(U*A.T))
print'Ap={}'.format(Ap)