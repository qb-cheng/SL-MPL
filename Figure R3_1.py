# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:08:35 2025

@author: QC
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

TrajLens = [151,301,451]
dts = [1,50]
ns_ratio = 0.01

Num_N = 4
Num_SC = 4
fig,axes = plt.subplots(Num_SC,Num_N,figsize=(15,3.5*Num_N),dpi=dpi_val)
for SC_idx in range(Num_SC):
    for N_idx in range(Num_N):
        thisSet = 3200 + N_idx*10 + SC_idx+1
        N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
        with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
            StoTrajs = pickle.load(f)
        ns = int(N * ns_ratio)
        SampledStoTrajs = Functions.SampleOnce(StoTrajs, ns)  
        
        estimates = []
        Ts = []
        steps = []
        cur_ax = axes[SC_idx,N_idx]
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
        if N_idx == 0:
            if SC_idx == 0:
                cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)  
            cur_ax.set_ylabel("Selection coefficient\nestimate $\^s$", fontsize=FontSize)
        else:
            cur_ax.set_ylabel("", fontsize=FontSize)
            cur_ax.axes.get_yaxis().set_ticklabels([])
        if SC_idx == Num_SC-1:
            cur_ax.set_xlabel("Trajectory length, $T$", fontsize=FontSize)  
        else:
            cur_ax.set_xlabel("", fontsize=FontSize)  
            cur_ax.axes.get_xaxis().set_ticklabels([])
        cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.set_ylim((-0.04,0.06))
        if SC_idx == 0:
            cur_ax.set_title('$N = $' + str(N), fontsize=FontSize) 
        
        
plt.tight_layout()
if save_flag:
    plt.savefig('./Figures/FigR3_1.jpg',dpi=dpi_val,bbox_inches='tight')   