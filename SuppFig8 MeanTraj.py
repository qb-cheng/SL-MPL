# -*- coding: utf-8 -*-
"""
Show that the mean trajectories, averaged over WF trajectories, is close to the
quasispecies trajectories (effectively with infinite N).
This allows approximating the sampling-only variance in the stochastic scenario
to the estimator variance in the deterministic scenario.
This also allows approximating the CRLB of the drift-only variance to 1/NV, 
where V represents the integrated variance of the quasispecies trajectory.
"""

import pickle,GetCaseArg,Functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


save_flag = True

dpi_val = 600

FigSizeR = 4
FontSize = 14
MarkerSize = 8
lw = 1.5


dt = 10

fig,axes = plt.subplots(3,3,figsize=(15,10),dpi=dpi_val)

for r_idx in range(3):
    for c_idx in range(3):
        Sets = c_idx*100 + r_idx + np.array([0,10,20,30])
        cur_ax = axes[r_idx,c_idx]
        for set_idx,thisSet in enumerate(Sets):
            N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
            t = np.arange(0,T,dt)
            with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
                StoTrajs = pickle.load(f)
                
            MeanTraj = np.mean(StoTrajs,axis=1)
            cur_ax.plot(t,MeanTraj[t],label='$N = $'+str(N),linewidth=lw)
            
        DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)
        cur_ax.plot(t,DeterTraj[t],marker='+',label='$N = \infty$',linewidth=lw)
            
        cur_ax.set_xlim((0,T+20))
        cur_ax.set_ylim((-0.05,1.05))        
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
        cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        cur_ax.text(0.95,0.1,'$s = $'+str(s)+', $x(0) = $'+str(x0),fontsize=FontSize,ha='right',va='center',transform=cur_ax.transAxes)
        
        if r_idx == 0 and c_idx == 0:
            cur_ax.legend(frameon=False,fontsize=FontSize,loc=2)
            
        if c_idx == 0:
            cur_ax.set_ylabel('Mean mutant allele frequency',fontsize=FontSize)
        else:
            cur_ax.axes.get_yaxis().set_ticklabels([])
            
        if r_idx == 2:
            cur_ax.set_xlabel('Generation, $t$',fontsize=FontSize)
        else:
            cur_ax.axes.get_xaxis().set_ticklabels([])
            

plt.tight_layout()
plt.subplots_adjust(wspace=0.5/15,hspace=0.5/10)


if save_flag:
    plt.savefig('./Figures/SuppFig8_MeanTraj.pdf',dpi=dpi_val,bbox_inches='tight')