# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:53:26 2025

@author: QC
"""

import Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns

save_flag = True

dpi_val = 600
FontSize = 14
MarkerSize = 5
lw = 1.5
Palette = sns.color_palette('Set3')

ns = 20
dt = 10
MCRun = 100000

thisSet = 4111

# Plot the sampling-only variance
fig,axes = plt.subplots(1,3,figsize=(15,4),dpi=dpi_val) 
    
# High mutation rate scenario
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
t = np.arange(0,T,dt)
DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)
DeterIV = Functions.ComputeIV(DeterTraj[t], dt)
SampledTrajs = Functions.SampleTrajectory(DeterTraj[t],ns,MCRun)
sFS = Functions.SLMPL(SampledTrajs,dt,ns,u)
EmpiVar = np.zeros(sFS.shape[0])
for t_idx,CurEst in enumerate(sFS):
    CurEst = CurEst[np.isfinite(CurEst)]
    EmpiVar[t_idx] = np.var(CurEst)
    
axes[0].plot(t,DeterTraj[t],linewidth=lw,color=Palette[2],label='$\mu = $'+f"{u:.0e}")
axes[1].plot(t[2:],DeterIV,linewidth=lw,color=Palette[2],label='$\mu = $'+f"{u:.0e}")
axes[2].semilogy(t[2:],EmpiVar,linewidth=lw,color=Palette[2],label='$\mu = $'+f"{u:.0e}")
    
    
# ow mutation rate scenario
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet+1)
t = np.arange(0,T,dt)
DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)
DeterIV = Functions.ComputeIV(DeterTraj[t], dt)
SampledTrajs = Functions.SampleTrajectory(DeterTraj[t],ns,MCRun)
sFS = Functions.SLMPL(SampledTrajs,dt,ns,u)
EmpiVar = np.zeros(sFS.shape[0])
for t_idx,CurEst in enumerate(sFS):
    CurEst = CurEst[np.isfinite(CurEst)]
    EmpiVar[t_idx] = np.var(CurEst)
    
cur_ax = axes[0]
cur_ax.plot(t,DeterTraj[t],linewidth=lw,color=Palette[3],label='$\mu = $'+f"{u:.0e}",linestyle='-.')
cur_ax.set_xlim((0,T+9))
cur_ax.set_xlabel('Generation, $t$',fontsize=FontSize)
cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)
cur_ax.set_ylim((0,1))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
cur_ax.set_ylabel('Mutant allele frequency, $x$',fontsize=FontSize)
# cur_ax.text(0.95,0.1,'$s = $'+str(s),fontsize=FontSize,ha='right',va='center',transform=cur_ax.transAxes)

cur_ax = axes[1]
cur_ax.plot(t[2:],DeterIV,linewidth=lw,color=Palette[3],label='$\mu = $'+f"{u:.0e}",linestyle='-.')
cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
cur_ax.set_ylabel('Integrated variance, $V$',fontsize=FontSize)
cur_ax.set_xlim((0,T+9))
cur_ax.set_ylim((0,30))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(10)) 
# cur_ax.text(0.95,0.1,'$s = $'+str(s),fontsize=FontSize,ha='right',va='center',transform=cur_ax.transAxes)

cur_ax = axes[2]
cur_ax.semilogy(t[2:],EmpiVar,linewidth=lw,color=Palette[3],label='$\mu = $'+f"{u:.0e}",linestyle='-.')
cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator variance, Var[$\\^s$]',fontsize=FontSize)
cur_ax.set_xlim((0,T+9))
cur_ax.set_ylim((1e-5,1e-2))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
# cur_ax.text(0.95,0.9,'$s = $'+str(s),fontsize=FontSize,ha='right',va='center',transform=cur_ax.transAxes)
    
    
plt.tight_layout()    
if save_flag:
    plt.savefig('./Figures/FigR4_1.jpg',dpi=dpi_val,bbox_inches='tight')