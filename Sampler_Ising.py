#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import time
from collections import Counter

t0 = time.time()

L = 5      # Grid size (LxL grid)
N = 10**6
Nconf = 2**(L*L) # Number of configurations
Tsteps = (L*L)

Ids = np.zeros((L, L, 2**(L*L))) 
C = list(itertools.product([0,1],repeat= L*L))
combinations_array = ((np.array(C))).astype(int)

# original_array = np.array([2**(L*L-i-1) for i in range(L*L)])
# twopow =  np.tile(np.transpose(original_array), (2**(L*L),1))
# twopow2 = np.tile(np.transpose(original_array), (N,1))
# label  =  np.sum(combinations_array * twopow,axis=1)


# Construct the set of configurations
#label them
# Construct the markov transition matrix


# Ising model
J = np.zeros((L*L,L*L))
h = np.random.choice([-0.05,0.05], L*L)

for x in range(L):
    for y in range(L):
        i = x * L + y  # Flattened index

        # Right neighbor with periodic boundary
        j = x * L + ((y + 1) % L)
        if i < j:  # Avoid double counting
            J[i, j] = J[j, i] = np.random.choice([-1.2, 1.2])

        # Bottom neighbor with periodic boundary
        j = ((x + 1) % L) * L + y
        if i < j:  # Avoid double counting
            J[i, j] = J[j, i] = np.random.choice([-1.2, 1.2])

#%%     
        
ismod= np.zeros(2**(L*L))

for i in range(0,2**(L*L)):
   Jxx = -np.dot((combinations_array[i,:]-1/2)*2,np.dot(J,(combinations_array[i,:]-1/2)*2))
   ismod[i] = np.exp(Jxx- np.dot(h,(combinations_array[i,:]-1/2)*2))
   
ismod = ismod/np.sum(ismod)   

cv = np.zeros((Tsteps,2**(L*L)))
cv_neu = np.zeros((Tsteps,2**(L*L)))
probn = np.zeros((Tsteps,2**(L*L)))

cv[0,:] = ismod
cv_neu[0,:] = cv[0,:]
probn[0,:] = ismod



# Initialize the samples as Ising model
Nc= 0

samples = np.zeros(((L*L),N))
samlabel0 = np.zeros(N)

samlabel = np.random.choice(np.arange(0,2**(L*L)), size=N, p=ismod)
samlabel0 = samlabel 

expon = L*L-np.arange(L*L)-1
samples = (samlabel[:,None] // (2 ** expon)) % 2 # converts labels to binary
samples = samples.T
#%%
filename = f"Edwards-Anderson_model_L{L}_samples.npy"
np.save(filename,samples)
filename = f"Edwards-Anderson_model_L{L}.npy"
np.save(filename,ismod)
