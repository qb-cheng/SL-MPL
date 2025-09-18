# -*- coding: utf-8 -*-
"""
Given simulation parameter settings, 
compute the empirical and analytical estimator variance (in the stochastic scenario) 
under the joint effect of limited sampling and genetic drift.
# Plot the estimator variance for different selection coefficients, and different trajectory lengths.
"""


import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
from scipy.interpolate import interp1d

save_flag = True
dpi_val = 600
FontSize = 14
MarkerSize = 3
lw = 1.5
Palette = sns.color_palette('Set1')

dt = 10
ns = 20
T_interests = [51,151,301]
N_indices = [0,1,2] # Different values of population sizes used to simulate the WF trajectories

x_values = np.linspace(0.0005,0.0525,1001)
EstVar_ref = (x_values/2)**2 
# Assuming that the estimates are normally distributed, 
# the empirical standard deviation should not exceed half of 
# the true selection coefficient in order to fit within the 5% significance level
xmin = 0.0005
xmax = 0.0525
ymin = 2e-6
ymax = 1e-2

fig,axes = plt.subplots(len(T_interests),len(N_indices),figsize=(15,4*len(T_interests)),dpi=dpi_val)
for T_idx,CurT in enumerate(T_interests):
    for N_idx in N_indices:
        CurSets = np.arange(600 + N_idx*10 + 1, 600 + N_idx*10 + 11)
        SCs = np.zeros(len(CurSets))
        EstVar_analytical = np.zeros(len(CurSets))
        EstVar_empirical = np.zeros(len(CurSets))

        for set_idx,thisSet in enumerate(CurSets):
            N,u,s,x0,_,_ = GetCaseArg.GetCaseInfo(thisSet)
            SCs[set_idx] = s
            t = np.arange(0,CurT,dt)
            
            with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
                StoTrajs = pickle.load(f)
            
            DeterTraj = Functions.DeterministicTrajectory(s,x0,CurT,u)
            DeterIV = Functions.ComputeIV(DeterTraj[t], dt)
            
            SampledStoTrajs = Functions.SampleOnce(StoTrajs[t], ns)
            sFS_joint = Functions.SLMPL(SampledStoTrajs, dt, ns, u)
            
            # Compute the empirical estimator variance under joint effects of finite sampling and genetic drift
            CurEsts = sFS_joint[-1,:]
            CurEsts = CurEsts[np.isfinite(CurEsts)]
            EstVar_empirical[set_idx] = np.var(CurEsts)
            
            #Compute the analytical approximate closed-form sampling-only and drift-only variance
            ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t],dt,ns,u)
            SamplingVar_analytical = ((VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2))[-1]
            DriftVar_analytical = 1 / (N * DeterIV[-1]);
            EstVar_analytical[set_idx] = SamplingVar_analytical + DriftVar_analytical

        
        cur_ax = axes[T_idx,N_idx]
        cur_ax.semilogy(SCs,EstVar_empirical,linewidth=lw,color=Palette[1],label='Empirical')
        cur_ax.semilogy(SCs,EstVar_analytical,linewidth=lw,color=Palette[0],linestyle='none',marker='o',ms=MarkerSize,label='Analytical')
        cur_ax.semilogy(x_values,EstVar_ref,linewidth=lw,linestyle='--',color='grey',label='5% significance level')
        
        interp_func = interp1d(SCs, EstVar_empirical, kind='linear', fill_value='extrapolate')
        EstVar_empirical_interpolated = interp_func(x_values)
        x_fill = x_values[EstVar_empirical_interpolated <= EstVar_ref]
        cur_ax.fill_between(x_fill, ymin, ymax, color='C2', alpha=0.1, label='Detectable range')
        
        cur_ax.set_xlim((xmin,xmax))
        cur_ax.set_ylim((ymin,ymax))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize) 
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
        
        if T_idx == 0:
            cur_ax.set_title('$N = $' + str(N), fontsize=FontSize, pad=10)
        if T_idx == len(T_interests)-1:
            cur_ax.set_xlabel('Selection coefficient, $s$',fontsize=FontSize)
        else:
            cur_ax.axes.get_xaxis().set_ticklabels([])
        if N_idx == 0:
            cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
            cur_ax.text(-0.35, 0.5, '$T = $' + str(CurT-1), ha='center', va='center', fontsize=FontSize, transform=cur_ax.transAxes)
            if T_idx == 0:
                cur_ax.legend(fontsize=FontSize,frameon=False,loc=4) 
        else:
            cur_ax.axes.get_yaxis().set_ticklabels([])

plt.tight_layout()
plt.subplots_adjust(wspace=0.5/12,hspace=0.5/12)

if save_flag:
    plt.savefig('./Figures/SuppFig12_PracticalImplication.pdf',dpi=dpi_val,bbox_inches='tight')