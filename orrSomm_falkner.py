#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:42:45 2019

@author: mfurquan
"""

import numpy as np
from numpy.polynomial import Chebyshev as Ch
from scipy.integrate import solve_bvp
import scipy.linalg as la
from math import pi
import matplotlib.pyplot as plt

"""
Input parameters:
"""
alpha = 0.2
beta = 0.0
Re = 500.0

N = 150 # Order of Chebyshev polynomials

H = 30.0
def transform(y):
    return -H*np.log(0.5*(1.0-y))

def Dy(y,n):
    return (1.0-y)/(-H)**n
""""""

""" Find velocity profile """
def Fprime(eta, F):
    return [F[1], F[2], -F[0]*F[2]-beta*(1.0-F[1]**2)]

def bc(Fa,Fb):
    return np.array([Fa[0],Fa[1],Fb[1]-1.0])

x = np.linspace(0.,20.,10)
f0 = np.zeros((3, x.size))
f = solve_bvp(Fprime, bc, x, f0)
""""""

""" Define Orr-Sommerfeld operators """
k2 = alpha**2 + beta**2
jaRe = 1.0j*alpha*Re

def orr_somm_lhs(T,y):
    eta = transform(y)
    U = f.sol(eta)[1]
    Upp = -f.sol(eta)[0]*f.sol(eta)[2]-beta*(1.0-f.sol(eta)[1]**2)
    v = T(y)
    v2 = Dy(y,1)**2*T.deriv(2)(y) + Dy(y,2)*T.deriv(1)(y)
    v4 = (    Dy(y,1)**4                       *T.deriv(4)(y)
       +  6.0*Dy(y,1)**2*Dy(y,2)               *T.deriv(3)(y)
       + (3.0*Dy(y,2)**2 + 4.0*Dy(y,2)*Dy(y,3))*T.deriv(2)(y)
       +      Dy(y,4)                          *T.deriv(1)(y))
    
    return ((-U*k2 - Upp -     k2**2 /jaRe)*v
          + ( U          + 2.0*k2    /jaRe)*v2
          + (            - 1.0       /jaRe)*v4)
          
def orr_somm_rhs(T,y):
    v = T(y)
    v2 = Dy(y,1)**2*T.deriv(2)(y) + Dy(y,2)*T.deriv(1)(y)
    return v2 - k2*v
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
        M[i,0]     = T(-1.0)
        M[i,1]     = T.deriv(1)(-1.0)*Dy(-1.0,1)
        M[i,2:N-1] = operator(T,y)
        M[i,N-1]   = T(1.0)
        M[i,N]     = T.deriv(1)(1.0)*Dy(1.0,1)
    M = eliminate(M,0)
    M = eliminate(M,1)
    M = eliminate(M,N-1)
    #M = eliminate(M,N)
    return M[2:N-1,2:N-1]
""""""
      
""" Get eigenvalues """      
A = get_Matrix(orr_somm_lhs)
B = get_Matrix(orr_somm_rhs)
c = la.eigvals(A,B)

plt.xlim(0.,1.2)
plt.ylim(-1.,0.)
plt.plot(c.real,c.imag,'o')