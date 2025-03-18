# -*- coding: utf-8 -*-
"""
Plot the estimator variance, sampling-only variance and drift-only variance
with different values of population sizes and sample sizes.
The cross-over points between the curves of sampling-only and drift-only variances
illustrate the moment that genetic drift effect becomes the dominant factor
over limited sampling.
"""

import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 


save_flag = True
dpi_val = 350
FontSize = 14
lw = 1.5



dt = 10
MCRun = 1000000

ymin = 1e-8
ymax = 1e-2


Sets = np.array([11,21,31,41,61])
ns_ratios = [0.005,0.01,0.05,0.1]

# Adder = np.array([[2,2,2,1],[2,1,2,1],[1,2,1,1],[1,1,1,1]])

fig,axes = plt.subplots(len(Sets),len(ns_ratios),figsize=(15,15),dpi=dpi_val)
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
    t = np.arange(0,T,dt)
    
    for ns_idx,ns_ratio in enumerate(ns_ratios):
        ns = int(N * ns_ratio)
        with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
            StoTrajs = pickle.load(f)
            
        # Compute the empirical estimator variance under joint effects of finite sampling and genetic drift
        JointVar = np.zeros(len(t)-2)
    
        SampledStoTrajs = Functions.SampleOnce(StoTrajs[t], ns)
        sFS_joint = Functions.SLMPL(SampledStoTrajs, dt, ns, u)
    
        for idx in range(sFS_joint.shape[0]):
            CurEsts = sFS_joint[idx,:]
            CurEsts = CurEsts[np.isfinite(CurEsts)]
            JointVar[idx] = np.var(CurEsts)
    
    
        # Compute the empirical sampling-only variance
        MeanTraj = np.mean(StoTrajs,1)
        SamplingVar_empirical = np.zeros(len(t)-2)
        SampledMeanTrajs = Functions.SampleTrajectory(MeanTraj[t], ns, MCRun)
        sFS_sampling = Functions.SLMPL(SampledMeanTrajs, dt, ns, u)
        for idx in range(sFS_sampling.shape[0]):
            CurEsts = sFS_sampling[idx,:]
            CurEsts = CurEsts[np.isfinite(CurEsts)]
            SamplingVar_empirical[idx] = np.var(CurEsts)
    
    
        # Compute the empirical drift-only variance
        DriftVar_empirical = np.var(Functions.SLMPL(StoTrajs[t], dt, 0, u),1)
    
        
    
        # Plot the empirical and analytical sampling-only and drift-only variance to 
        # show that our analytical approximation works
        cur_ax = axes[set_idx,ns_idx]
        cur_ax.semilogy(t[2:],JointVar,label='Var$[\\^s]$',color='C1',linewidth=lw,zorder=2,linestyle='--')
        cur_ax.semilogy(t[2:],SamplingVar_empirical,label=r'$\sigma_{\mathrm{s}}^2[\^s]$',color='C0',linewidth=lw,zorder=1)
        cur_ax.semilogy(t[2:],DriftVar_empirical,label=r'$\sigma_{\mathrm{d}}^2[\^s]$',color='C2',linewidth=lw,zorder=1)
        # if np.sum(SamplingVar_empirical > DriftVar_empirical) < len(SamplingVar_empirical):
        #     cur_ax.axvline(t[np.sum(SamplingVar_empirical > DriftVar_empirical)+2],color='gray',linewidth=lw,linestyle='--')
        
        
        if ns_idx == 0:
            cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
            cur_ax.text(-0.5, 0.5, '$N = $' + str(N), ha='center', va='center', fontsize=FontSize, transform=cur_ax.transAxes)
            if set_idx == 0:
                cur_ax.legend(fontsize=FontSize,frameon=False)
        else:
            cur_ax.axes.get_yaxis().set_ticklabels([])
        
        
        if set_idx == 0:
            cur_ax.set_title('$n_s$ / $N = $' + str(ns_ratio), fontsize=FontSize, pad=20)
            
        if set_idx == len(Sets)-1:
            cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
        else:
            cur_ax.axes.get_xaxis().set_ticklabels([])
                
        cur_ax.set_xlim((0,T+25))
        cur_ax.set_ylim((ymin, ymax))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
        # cur_ax.text(0.2,0.05,'$N$ / $n_s = $' + str(int(N/ns)),ha='center',fontsize=FontSize, transform=cur_ax.transAxes)
        # cur_ax.text(0.23,0.05,'$n_s$ = ' + str(ns),ha='center',fontsize=FontSize, transform=cur_ax.transAxes)
        
        


plt.tight_layout()
plt.subplots_adjust(wspace=0.1,hspace=0.1)

if save_flag:
    plt.savefig('./Figures/SuppFig9_CrossOver.jpg',dpi=dpi_val,bbox_inches='tight')  