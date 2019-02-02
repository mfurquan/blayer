#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare solutions of Falkner-Skan eqn. obtained using
shooting method and spline collocation

Created on Sun Jan 27 23:24:12 2019

@author: mfurquan

Eqn:  f"'(eta) + f(eta)f"(eta) + beta*(1 - f'(eta)^2)
BCs:  f(0) = f'(0) = 0, f'(infinity) = 1
"""
from scipy.integrate import solve_ivp, solve_bvp
import matplotlib.pyplot as plt
import numpy as np

beta = 0.
tol = 1.e-8
maxiter = 10
eps = 1.e-4
eta_max = 2.0

# define as first order system
def Fprime(eta, F):
    return [F[1], F[2], -F[0]*F[2]-beta*(1.0-F[1]**2)]

"""
Using shooting method (Newton) with solve_ivp
"""
# Newton-Raphson procedure
def newton(f,guess):
    def newton_itr(x):
        return x - f(x)*eps/(f(x+eps)-f(x))
    for i in range(maxiter):
        guess = newton_itr(guess)
        if abs(f(guess))<tol:
            break
        else:
            print("iteration no.:",i," off by: ",f(guess))
    return guess

# (f'(1)-1) as function of f"(0)
def F2max(F3_0):
    global eta_max# = 2.0
    diff = 1.0
    while diff > tol:
        F       = solve_ivp(Fprime,[0,eta_max],[0.,0.,F3_0],t_eval=[eta_max])
        Fplus   = solve_ivp(Fprime,[0,eta_max+1.0],[0.,0.,F3_0],t_eval=[eta_max])
        diff    = abs(Fplus.y[1,0]-F.y[1,0])
        eta_max = eta_max + 1.0
    print("eta_max=",eta_max)
    return F.y[1,0]-1.0

F3_0 = newton(F2max,0.3)
print("f\"\'(0)=",F3_0)
F = solve_ivp(Fprime,[0,10.0],[0.,0.,F3_0])

"""
Using solve_bvp
"""
def bc(Fa,Fb):
    return np.array([Fa[0],Fa[1],Fb[1]-1.0])

x = np.linspace(0.,10.,10)
y = np.zeros((3, x.size))
sol = solve_bvp(Fprime, bc, x, y)

"""
Plot the two solutions
"""
x_plot = np.linspace(0, 10, 100)

plt.plot(F.t,F.y[0,:],'ro',label="solve_ivp")
y_plot = sol.sol(x_plot)[0]
plt.plot(x_plot, y_plot,'b-',label="solve_bvp")
plt.ylabel('$f(\eta)$')
plt.xlabel('$\eta$')
plt.legend()
plt.show()

plt.plot(F.t,F.y[1,:],'ro',label="solve_ivp")
y_plot = sol.sol(x_plot)[1]
plt.plot(x_plot, y_plot,'b-',label="solve_bvp")
plt.ylabel('$f\'(\eta)$')
plt.xlabel('$\eta$')
plt.legend()
plt.show()

plt.plot(F.t,F.y[2,:],'ro',label="solve_ivp")
y_plot = sol.sol(x_plot)[2]
plt.plot(x_plot, y_plot,'b-',label="solve_bvp")
plt.ylabel('$f\'\'\'(\eta)$')
plt.xlabel('$\eta$')
plt.legend()
plt.show()