# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 19:44:00 2025

@author: QC
"""

import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns


save_flag = True

dpi_val = 600
FontSize = 16
lw = 1.5
Palette = sns.color_palette('tab10')

xmin = -0.04
xmax = 0.08
bins = np.linspace(xmin,xmax,50)


Sets = [11,41]

T = 151
# dts = np.array([factor for factor in range (1,T) if (T-1) % factor == 0 and factor <= (T-1)/3])
dts = [1,50,75]
ns = 20

fig,axes = plt.subplots(len(dts),len(Sets),figsize=(15,len(dts)*4),dpi=dpi_val) 
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,_ = GetCaseArg.GetCaseInfo(thisSet)
    with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
        StoTrajs = pickle.load(f)
        StoTrajs = StoTrajs[:T]
        
    SampledStoTrajs = Functions.SampleOnce(StoTrajs, ns)  

    for dt_idx,dt in enumerate(dts):
        t = np.arange(0,T,dt)
        
        # The joint variance w.r.t. dt in the stochastic evolutionary scenario
        ObservedSampledStoTrajs = SampledStoTrajs[t]
        D_hat = ObservedSampledStoTrajs[-1,:] - ObservedSampledStoTrajs[0,:] - u*dt*np.sum(1-2*ObservedSampledStoTrajs[:-1,:],axis=0)
        V_hat = dt * np.sum(ObservedSampledStoTrajs[:-1,:]*(1-ObservedSampledStoTrajs[:-1,:]),axis=0) / (1-1/ns)
        s_FS = D_hat / V_hat
        s_FS = s_FS[np.isfinite(s_FS)]
        print(np.mean(s_FS <= 0)*100)
        
        cur_mean = np.mean(s_FS)
        cur_sigma = np.std(s_FS)
        
        cur_ax = axes[dt_idx,set_idx]
        cur_ax.hist(s_FS,bins=bins,alpha=0.4,density=True,edgecolor='black',linewidth=lw/2,color='C'+str(dt_idx))
        cur_ax.axvline(x = cur_mean-cur_sigma, color ="gray", linestyle = "dashed",linewidth=lw)
        cur_ax.axvline(x = cur_mean+cur_sigma, color ="gray", linestyle = "dashed",linewidth=lw)
        cur_ax.axvline(x = cur_mean-2*cur_sigma, color ="gray", linestyle ="dashed",linewidth=lw)
        cur_ax.axvline(x = cur_mean+2*cur_sigma, color ="gray", linestyle ="dashed",linewidth=lw)
        cur_ax.axvline(x = cur_mean-3*cur_sigma, color ="gray", linestyle ="dashed",linewidth=lw)
        cur_ax.axvline(x = cur_mean+3*cur_sigma, color ="gray", linestyle ="dashed",linewidth=lw)
        cur_ax.set_xlim((xmin,xmax))
        cur_ax.set_ylim((0,150))
        if dt_idx == len(dts)-1:
            cur_ax.set_xlabel('Selection coefficient estimate, $\\^s$',fontsize=FontSize)
        else:
            cur_ax.set_xlabel('',fontsize=FontSize)
            cur_ax.axes.get_xaxis().set_ticklabels([])
        if dt_idx == 0:
            cur_ax.set_title('$N = $'+str(N),fontsize=FontSize)
        if set_idx == 0:
            cur_ax.set_ylabel('Density',fontsize=FontSize)
            cur_ax.text(-0.3,0.5,'$\Delta t = $' + str(dt),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)
        else:
            cur_ax.set_ylabel('',fontsize=FontSize)
            cur_ax.axes.get_yaxis().set_ticklabels([])
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
        cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(50))


plt.tight_layout()
plt.subplots_adjust(wspace=0.1,hspace=0.15)
    
if save_flag:
    plt.savefig('./Figures/FigR3_2.jpg',dpi=dpi_val,bbox_inches='tight')