#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:12:12 2019

@author: mfurquan
"""

from numpy.polynomial import Chebyshev as Ch
T = Ch.fit([-1,1],[-1,1],1)
def op(S):
    return S**2
