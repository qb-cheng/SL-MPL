# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:41:33 2025

@author: cerul
"""

import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import pandas as pd
import seaborn as sns


save_flag = True

dpi_val = 600
FontSize = 16
lw = 1.5

nss_default = [20,100]
dts = [1,50]
TrajLens = [151,301,451]

Sets = np.array([2102,2302,2202,2402])
fig,axes = plt.subplots(len(Sets),len(nss_default)+1,figsize=(15,3*len(Sets)),dpi=dpi_val) 
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
    with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
        StoTrajs = pickle.load(f)
    nss = nss_default.copy()
    nss.append(N)
    
    for ns_idx,ns in enumerate(nss):
        if ns < N:
            SampledStoTrajs = Functions.SampleOnce(StoTrajs, ns)  
        else:
            SampledStoTrajs = StoTrajs
        estimates = []
        Ts = []
        steps = []
        cur_ax = axes[set_idx,ns_idx]
        
        for TrajLen_idx,TrajLen in enumerate(TrajLens):
            for dt_idx,dt in enumerate(dts):
                t = np.arange(0,TrajLen,dt)
                ObservedSampledStoTrajs = SampledStoTrajs[t]
                D_hat = ObservedSampledStoTrajs[-1,:] - ObservedSampledStoTrajs[0,:] - u*dt*np.sum(1-2*ObservedSampledStoTrajs[:-1,:],axis=0)
                V_hat = dt * np.sum(ObservedSampledStoTrajs[:-1,:]*(1-ObservedSampledStoTrajs[:-1,:]),axis=0) / (1-1/ns)
                s_FS = D_hat / V_hat
                s_FS = s_FS[np.isfinite(s_FS)]
                estimates += list(s_FS)
                Ts += [TrajLen-1]*len(s_FS)
                steps += ['$\Delta t$ = ' + str(dt)]*len(s_FS)
            
        df_est = pd.DataFrame()
        df_est['Estimate'] = estimates
        df_est['TrajLens'] = Ts
        df_est['TimeSamplingStep'] = steps
        sns.boxplot(x="TrajLens",y="Estimate",hue="TimeSamplingStep",data=df_est,width=0.5,dodge=True,fliersize=0,ax=cur_ax,flierprops={"marker": "o"})
        cur_ax.axhline(y = s, color ='gray', linestyle ="dotted",linewidth=lw)
        cur_ax.legend_.remove()
        cur_ax.text(0.05,0.9,'$N = $' + str(N),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)
        if ns_idx == 0:
            if set_idx == 0:
                cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)    
            cur_ax.set_ylabel("Selection coefficient\nestimate $\^s$", fontsize=FontSize)
        else:
            cur_ax.set_ylabel("", fontsize=FontSize)
            cur_ax.axes.get_yaxis().set_ticklabels([])
        if set_idx == len(Sets)-1:
            cur_ax.set_xlabel("Trajectory length, $T$", fontsize=FontSize)  
        else:
            cur_ax.set_xlabel("", fontsize=FontSize)  
            cur_ax.axes.get_xaxis().set_ticklabels([])
        cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.set_ylim((-0.03,0.05))
        if set_idx == 0:
            if ns < N:
                cur_ax.set_title('$n_s = $' + str(ns), fontsize=FontSize)  
            else:
                cur_ax.set_title('$n_s = N$', fontsize=FontSize)  
        
        
plt.tight_layout()
if save_flag:
    plt.savefig('./Figures/FigR1_1.jpg',dpi=dpi_val,bbox_inches='tight')   