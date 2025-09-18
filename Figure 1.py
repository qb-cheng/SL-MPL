# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:25:17 2025

@author: QC
"""

import GetCaseArg,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns

save_flag = True
dpi_val = 600
FontSize = 16
lw = 1.5
Palette = sns.color_palette('Set2')
MarkerSize = 6

thisSet = 11
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)

dt = 30
t = np.arange(0,T,dt)

lambda_ns = 20
TrajIdx_Example = 27
TrajNum = 10000

with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
    StoTrajs = pickle.load(f)
StoTrajs = StoTrajs[:,:TrajNum]

SampleSizes = np.zeros((len(t),TrajNum))
ObservedTrajs = np.zeros((len(t),TrajNum))
for traj_idx in range(TrajNum):
    for t_idx,cur_t in enumerate(t):
        cur_ns = np.random.poisson(lambda_ns)
        while cur_ns < 2:
            cur_ns = np.random.poisson(lambda_ns)
        ObservedTrajs[t_idx,traj_idx] = np.random.binomial(cur_ns, StoTrajs[cur_t,traj_idx]) / cur_ns
        SampleSizes[t_idx,traj_idx] = cur_ns
        

fig,axes = plt.subplots(1,2,figsize=(10,4),dpi=dpi_val)
cur_ax = axes[0]
cur_ax.plot(np.arange(T),StoTrajs[:,TrajIdx_Example],linewidth=lw,color='gray',label='Population, $x$')
CurObTraj = ObservedTrajs[:,TrajIdx_Example]
cur_ax.plot(t,CurObTraj,linewidth=lw,label='Observations, $\hat{x}$',color=Palette[1],zorder=1,marker='.',ms=MarkerSize)

cur_ax.plot([t[8],t[8]],[0.6,CurObTraj[8]],linewidth=lw,linestyle='--',color='k')
cur_ax.plot([t[9],t[9]],[0.6,CurObTraj[9]],linewidth=lw,linestyle='--',color='k')

cur_ax.set_xlim((-5,T+9))
cur_ax.set_ylim((-0.05,1.05))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)   
cur_ax.set_xlabel('Generation, $t$',fontsize=FontSize)
cur_ax.set_ylabel('Mutant allele frequency',fontsize=FontSize)
cur_ax.legend(fontsize=FontSize,loc=4,frameon=False)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  
    

cur_ax = axes[1]
sFS = np.zeros(TrajNum)
for traj_idx in range(TrajNum):
    CurObTraj = ObservedTrajs[:,traj_idx]
    D = CurObTraj[-1] - CurObTraj[0] - u*dt*np.sum(1-2*CurObTraj[:-1])
    V = dt*np.sum(SampleSizes[:-1,traj_idx]/(SampleSizes[:-1,traj_idx]-1)*CurObTraj[:-1]*(1-CurObTraj[:-1]))
    sFS[traj_idx] = D / V

# bins = np.linspace(-0.03,0.07,50)   
# cur_ax.hist(sFS,bins=bins,density=True,color=Palette[1],edgecolor='black',linewidth=lw/2)
# cur_ax.set_xlim((bins[0],bins[-1]))
sns.violinplot(sFS,inner="quart", linewidth=lw, ax=cur_ax, color=Palette[1], cut=0, alpha=0.8, scale='area')
cur_ax.set_ylim((0,0.06))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)  
cur_ax.set_xticks([])
cur_ax.set_xlabel('Sampling and drift\neffects on estimator',fontsize=FontSize) 
cur_ax.set_ylabel('Selection coefficient\nestimate, $\hat{s}$',fontsize=FontSize)
# cur_ax.set_ylabel('Density',fontsize=FontSize)
# cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02)) 

plt.tight_layout()
x_pos = axes[0].get_position().intervalx
y_pos = axes[0].get_position().intervaly
axes[0].set_position([x_pos[0], y_pos[0], x_pos[1]-x_pos[0]+0.12, y_pos[1]-y_pos[0]])

x_pos = axes[1].get_position().intervalx
y_pos = axes[1].get_position().intervaly
# axes[1].set_position([x_pos[0]+0.15, y_pos[0]+0.18, x_pos[1]-x_pos[0]-0.15, y_pos[1]-y_pos[0]-0.2])
axes[1].set_position([x_pos[0]+0.14, y_pos[0], x_pos[1]-x_pos[0]-0.14, y_pos[1]-y_pos[0]])

if save_flag:
    plt.savefig('./Figures/Fig1_IntroFig.jpg',dpi=dpi_val,bbox_inches='tight')  
    # plt.savefig('./Figures/Fig1_IntroFig.pdf',dpi=dpi_val,bbox_inches='tight')  