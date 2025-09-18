# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 00:46:30 2025

@author: QC
"""

# Study the estimator performance (mean and variance) with respect to delta t
# Assuming that the trajectory has enough informative samples (i.e., away from the boundary),
# how the sampling-only and drift-only variance change with dt, for a fixed T?

import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns


save_flag = True

dpi_val = 600
FontSize = 14
MarkerSize = 5
lw = 1.5
Palette = sns.color_palette('Set2')

ns = 20

# var_max = np.max([np.max(EmpiVar_sampling),np.max(EmpiVar_drift)])
# var_max_log = np.floor(np.log10(var_max))
# if var_max / (10**var_max_log) <= 5:
#     var_max = 5 * (10**var_max_log)
# else:
#     var_max = 10 ** (var_max_log+1)

Sets = np.array([np.arange(2102,2106),np.arange(2202,2206),np.arange(2502,2506)]) # Default N = 1000
xstep = 10 # 30 40 
T = 151
dts = np.array([factor for factor in range (1,T) if (T-1) % factor == 0 and factor <= (T-1)/3])

fig,axes = plt.subplots(Sets.shape[0],Sets.shape[1],figsize=(15,3.75*Sets.shape[0]),dpi=dpi_val) 
for row_idx in range(Sets.shape[0]):
    for col_idx in range(Sets.shape[1]):
        thisSet = Sets[row_idx,col_idx]
        N,u,s,x0,NumItr,_ = GetCaseArg.GetCaseInfo(thisSet)
        
        # Stochastic evolutionary scenario
        with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
            StoTrajs = pickle.load(f)
            StoTrajs = StoTrajs[:T]
            
        SampledStoTrajs = Functions.SampleOnce(StoTrajs, ns)  
        EmpiMean_joint = np.zeros(len(dts))
        for dt_idx,dt in enumerate(dts):
            t = np.arange(0,T,dt)
            
            # The joint mean w.r.t. dt in the stochastic evolutionary scenario
            ObservedSampledStoTrajs = SampledStoTrajs[t]
            D_hat = ObservedSampledStoTrajs[-1,:] - ObservedSampledStoTrajs[0,:] - u*dt*np.sum(1-2*ObservedSampledStoTrajs[:-1,:],axis=0)
            V_hat = dt * np.sum(ObservedSampledStoTrajs[:-1,:]*(1-ObservedSampledStoTrajs[:-1,:]),axis=0) / (1-1/ns)
            s_FS = D_hat / V_hat
            s_FS = s_FS[np.isfinite(s_FS)]
            EmpiMean_joint[dt_idx] = np.mean(s_FS)

        # Show that the joint mean is close to the true selection coefficient for various dt
        cur_ax = axes[row_idx,col_idx]
        cur_ax.plot(dts,EmpiMean_joint,label='$\\mathrm{E}[\\^s]$',color=Palette[0],linestyle='None',marker='o',ms=MarkerSize,markeredgewidth=2,markerfacecolor='w',linewidth=lw,zorder=2) # (,linestyle='dashdot', $n_s = $'+str(ns)+')
        cur_ax.axhline(y = s, color ="gray", linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$',zorder=1)
        if col_idx == 0:
            if row_idx == 0:
                cur_ax.legend(fontsize=FontSize,frameon=False)
            cur_ax.set_ylabel('Estimator mean\n($N = $' + str(N) + ')',fontsize=FontSize)
        else:
            cur_ax.axes.get_yaxis().set_ticklabels([])
        
        cur_ax.set_xlim((0,np.max(dts) + 2))
        cur_ax.set_ylim((-0.01,0.03))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
        cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        if row_idx == 0:
            cur_ax.set_title('$s = $'+str(s),fontsize=FontSize)            
        if row_idx == Sets.shape[0]-1:
            cur_ax.set_xlabel('Time sampling step, $\Delta t$',fontsize=FontSize)
        else:
            cur_ax.axes.get_xaxis().set_ticklabels([])
        cur_ax.text(0.2, 0.1, '$\Delta t = 1: \^s = $' + f"{EmpiMean_joint[0]:.2e}\n" 
                    + '$\Delta t = 50: \^s = $' + f"{EmpiMean_joint[-1]:.2e}\n"
                    + 'Change $\\approx$: ' + str(np.abs(np.int32(np.round((EmpiMean_joint[-1] - EmpiMean_joint[0]) / EmpiMean_joint[0]*100)))) + '%', ha='left', va='center', fontsize=FontSize, transform=cur_ax.transAxes)

plt.tight_layout()

if save_flag:
    plt.savefig('./Figures/FigR4_2.jpg',dpi=dpi_val,bbox_inches='tight')
    