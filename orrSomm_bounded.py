#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:29:27 2019

@author: mfurquan

Finds eigenvalues of Orr-Sommerfeld equation for plane Couette/Poiseuille flows
"""

import numpy as np
from numpy.polynomial import Chebyshev as Ch
import scipy.linalg as la
from math import pi
import matplotlib.pyplot as plt

"""
Input parameters:
"""
profile_switch = 1 # 0: Couette Flow, 1 : Poiseuille Flow

alpha = 1.0 # streamwise wavenumber
beta = 0.0 # spanwise wavenumber
Re = 10000.0 # Reynold's number

N = 150 # Order of Chebyshev polynomials
""""""

""" Fit velocity profile """
if profile_switch==0:
    u = np.array([[-1.,1.],[-1.,1.]])
elif profile_switch==1:
    u = np.array([[-1.,0.,1.],[0.,1.,0.]])

U = Ch.fit(u[0,:],u[1,:],u.shape[1]-1)
""""""

""" Define Orr-Sommerfeld operators """
k2 = alpha**2 + beta**2
jaRe = 1.0j*alpha*Re

def orr_somm_lhs(T):
    return ((-U*k2 - U.deriv(2) -     k2**2 /jaRe)*T 
          + ( U                 + 2.0*k2    /jaRe)*T.deriv(2)
          + (                   - 1.0       /jaRe)*T.deriv(4))
    
def orr_somm_rhs(T):
    return (T.deriv(2) - k2*T)
""""""

""" Discretize Operator """
def eliminate(M,n):
    for i in range(0,N+1):
        if i!=n:
            M[:,i] = M[:,i] - M[:,n]*M[n,i]/M[n,n]
    return M

def get_Matrix(operator):
    M = np.zeros((N+1,N+1),dtype=complex)
    I = np.eye(N+1)
    y = np.cos(np.linspace(2,N-2,N-3,dtype=float)*pi/N)
    for i in range(0,N+1):
        T = Ch(I[i,:])
        M[i,0]     = T(1.0)
        M[i,1]     = T.deriv(1)(1.0)
        M[i,2:N-1] = operator(T)(y)
        M[i,N-1]   = T(-1.0)
        M[i,N]     = T.deriv(1)(-1.0)
    M = eliminate(M,0)
    M = eliminate(M,1)
    M = eliminate(M,N-1)
    M = eliminate(M,N)
    return M[2:N-1,2:N-1]
""""""
      
""" Get eigenvalues """      
A = get_Matrix(orr_somm_lhs)
B = get_Matrix(orr_somm_rhs)
c = la.eigvals(A,B)

plt.xlim(0.,1.)
plt.ylim(-1.,0)
plt.plot(c.real,c.imag,'o')