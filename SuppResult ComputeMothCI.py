# -*- coding: utf-8 -*-
"""
Estimate the selection coefficient of the moth data using the MPL-based estimator.
Compute the 95% confidence interval using the likelihood obtained in MPL.
"""

import numpy as np
import pandas as pd



def LogLikelihood(s,Traj,Years,N,u):
    """
    Compute the log-likelihood of observing a trajectory in MPL. 
    Note that we assume negligible sampling noise to obtain the likelihood.
    We remove observed frequencies that are exactly at the boundary (i.e., x = 0 in this case)
    to avoid numerical issues while computing the likelihood.
    ----------
    s : selection coefficient
    Traj : observed frequency trajectory
    Years : observable generation / calendar year
    N : (effective) population size which is assumed constant over time
    u : mutation probability
    """
    
    # Remove time points with zero observed frequencies (at the boundary)
    # np.where(Traj == 0)[0]
    NonZeroTraj = np.delete(Traj, np.where(Traj == 0)[0])
    NonZeroYears = np.delete(Years, np.where(Traj == 0)[0])
        
            
    dx = 1/N # Constant accounting for quantization 
    LL = 0
    for t_idx in range(len(NonZeroTraj)-1):
        cur_x = NonZeroTraj[t_idx]
        next_x = NonZeroTraj[t_idx+1]
        dt = NonZeroYears[t_idx+1] - NonZeroYears[t_idx]
        
        cur_v = cur_x * (1-cur_x)
        drift = dt * (s*cur_v + u*(1-2*cur_x))
        Theta = (next_x - cur_x - drift)**2 / (dt * cur_v)
        
        TransProb = dx * np.sqrt(N/(2*np.pi*dt*cur_v)) * np.exp(-N/2 * Theta)
        LL += np.log(TransProb)
        
    return LL



# Plot mutant allele frequency trajectories and sample size
df = pd.read_csv('./Moth Frequency.csv')

# Traj = df['Mutant'] / (df['Typical'] + df['Mutant'])
# Traj = np.array(Traj)
Traj = np.array(df['Allele freq'])

t = np.array(df['Year'])
t = t - t[0]
dt = t[1:] - t[:-1]
nss = np.array(df['Number captured']) * 2 # Multiplied by 2 because moth is diploid


Ds = np.zeros(len(t)-2)
Vs = np.zeros(len(t)-2)

for idx in range(len(t)-2):
    ObservableTraj = Traj[:idx+3]
    cur_ns = nss[:idx+2]
    
    Ds[idx] = ObservableTraj[-1] - ObservableTraj[0]
    Vs[idx] = np.sum((t[1:idx+3]-t[:idx+2]) * (ObservableTraj[:-1]*(1-ObservableTraj[:-1])) * (cur_ns/(cur_ns-1)))
    
ests = Ds / Vs




SCs = np.arange(-0.15,0.051,1e-3)

# For data up to 1995
ObservedTraj = Traj[:-4]
ObservedYears = np.array(df['Year'])[:-4]
LLs_1995 = np.zeros(len(SCs))
for sc_idx,cur_sc in enumerate(SCs):
    LLs_1995[sc_idx] = LogLikelihood(cur_sc,ObservedTraj,ObservedYears,1000,0)

# Computing the confidence interval using LR testing principle: the log-likelihood drop-off is no more than 1.92
# https://personal.psu.edu/abs12/stat504/Lecture/lec3_4up.pdf (page 37)
SC_interval_values_1995 = np.round(SCs[np.where(LLs_1995 >= np.max(LLs_1995) - 1.92)[0]],3)
print('Data from 1939-1995: MPL-based estimate is ' + str(np.round(ests[-5],3)) + ', with 95% CI of (' + str(SC_interval_values_1995[0]) + ', ' + str(SC_interval_values_1995[-1]) + ')')

# For data up to 1999
ObservedTraj = Traj
ObservedYears = np.array(df['Year'])
LLs_1999 = np.zeros(len(SCs))
for sc_idx,cur_sc in enumerate(SCs):
    LLs_1999[sc_idx] = LogLikelihood(cur_sc,ObservedTraj,ObservedYears,1000,0)

SC_interval_values_1999 = np.round(SCs[np.where(LLs_1999 >= np.max(LLs_1999) - 1.92)[0]],3)
print('Data from 1939-1999: MPL-based estimate is ' + str(np.round(ests[-1],3)) + ', with 95% CI of (' + str(SC_interval_values_1999[0]) + ', ' + str(SC_interval_values_1999[-1]) + ')')





