# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 23:22:34 2025

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
FontSize = 14
MarkerSize = 4
lw = 1.5

Palette = sns.color_palette('Set2')
Palette = [Palette[1], Palette[3], Palette[4]]

dt = 10
nss = [20,100]
Sets = np.arange(3101,3106)

fig,axes = plt.subplots(1,1,figsize=(8,4),dpi=dpi_val)

estimates = []
TrueSCs = []
SCs = []
SampleSizes = []
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
    SCs.append(s)
    t = np.arange(0,T,dt)

    with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
        StoTrajs = pickle.load(f)
        
    for ns_idx,ns in enumerate(nss):
        if ns != N:
            SampledStoTrajs = Functions.SampleOnce(StoTrajs[t], ns)
    
        D_hat = SampledStoTrajs[-1,:] - SampledStoTrajs[0,:] - u*dt*np.sum(1-2*SampledStoTrajs[:-1,:],axis=0)
        V_hat = dt * np.sum(SampledStoTrajs[:-1,:]*(1-SampledStoTrajs[:-1,:]),axis=0) / (1-1/ns)
        s_FS = D_hat / V_hat
        s_FS = s_FS[np.isfinite(s_FS)]
    
        estimates += list(s_FS)
        TrueSCs += [f"{s:.0e}"] * len(s_FS)
        if ns != N:
            SampleSizes += ['$n_s = $' + str(ns)] * len(s_FS)
        else:
            SampleSizes += ['$n_s = N$'] * len(s_FS)
    
cur_ax = axes
df_est = pd.DataFrame()
df_est['Estimate'] = estimates
df_est['TrueSC'] = TrueSCs
df_est['SampleSize'] = SampleSizes
sns.boxplot(x="TrueSC",y="Estimate",hue="SampleSize",data=df_est,width=0.35,dodge=True,fliersize=0,ax=cur_ax,flierprops={"marker": "o"},palette=Palette)
for SC_idx,SC in enumerate(SCs):
    cur_ax.plot([SC_idx-0.3,SC_idx+0.3],[SC,SC],color='C3',linewidth=lw,linestyle='--')
cur_ax.set_xlabel('Selection coefficient, $s$',fontsize=FontSize)
cur_ax.set_ylabel('Selection coefficient estimate, $\^s$',fontsize=FontSize)
cur_ax.set_ylim((-0.02,0.17))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05)) 
cur_ax.legend(fontsize=FontSize,frameon=False,loc=2)

plt.tight_layout()


if save_flag:
    plt.savefig('./Figures/FigR3_3.jpg',dpi=dpi_val,bbox_inches='tight')