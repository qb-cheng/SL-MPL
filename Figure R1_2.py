# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 15:42:21 2025

@author: QC
"""

import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 

save_flag = True
dpi_val = 600
FontSize = 14
MarkerSize = 4
lw = 1.5

dt = 10
MCRun = 1000000

ymin = 1e-7
ymax = 1e-2

Sets = np.array([2102,2302,2202,2402])
nss = [20,100]
fig,axes = plt.subplots(len(Sets),2,figsize=(10,3*len(Sets)),dpi=dpi_val)
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
    t = np.arange(0,T,dt)
    DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)
    DeterIV = Functions.ComputeIV(DeterTraj[t], dt)
    
    with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
        StoTrajs = pickle.load(f)
    MeanTraj = np.mean(StoTrajs,1)
    
    # Compute the empirical drift-only variance
    DriftVar_empirical = np.var(Functions.SLMPL(StoTrajs[t], dt, 0, u),1)
    
    # Compute the empirical CRLB
    StoIVs = np.zeros((len(t)-2,StoTrajs.shape[1]))
    for ItrIdx in range(StoTrajs.shape[1]):
        CurStoTraj = StoTrajs[:,ItrIdx]
        StoIVs[:,ItrIdx] = Functions.ComputeIV(CurStoTraj[t], dt)   
    StoIVs_mean = np.mean(StoIVs,axis=1)
    CRLB = 1 / (N*(StoIVs_mean))
    DriftVar_analytical = 1 / (N * DeterIV); 
    
    
    for ns_idx,ns in enumerate(nss):
        SampledStoTrajs = Functions.SampleOnce(StoTrajs[t], ns)
        sFS_joint = Functions.SLMPL(SampledStoTrajs, dt, ns, u)
        # Compute the empirical estimator variance
        JointVar_empirical = np.zeros(len(t)-2)
        for idx in range(sFS_joint.shape[0]):
            CurEsts = sFS_joint[idx,:]
            CurEsts = CurEsts[np.isfinite(CurEsts)]
            JointVar_empirical[idx] = np.var(CurEsts)
            
        # Compute the empirical sampling-only variance
        SamplingVar_empirical = np.zeros(len(t)-2)
        SampledMeanTrajs = Functions.SampleTrajectory(MeanTraj[t], ns, MCRun)
        sFS_sampling = Functions.SLMPL(SampledMeanTrajs, dt, ns, u)
        for idx in range(sFS_sampling.shape[0]):
            CurEsts = sFS_sampling[idx,:]
            CurEsts = CurEsts[np.isfinite(CurEsts)]
            SamplingVar_empirical[idx] = np.var(CurEsts)

 
        # Plot the empirical and analytical sampling-only and drift-only variance to 
        # show that our analytical approximation works
        cur_ax = axes[set_idx,ns_idx]
        line1 = cur_ax.semilogy(t[2:],JointVar_empirical,color='C8',linewidth=lw,zorder=2,linestyle='--')
        SumVar = SamplingVar_empirical + DriftVar_empirical
        InterestingLengths = np.arange(50,T,50) # np.array([50,100,150,300,450])
        InterestingEstVar = np.zeros(len(InterestingLengths))
        for T_idx,T_interest in enumerate(InterestingLengths):
            InterestingEstVar[T_idx] = SumVar[np.where(t == T_interest)[0][0] - 2]
        line2 = cur_ax.semilogy(InterestingLengths,InterestingEstVar,color='c',linestyle='None',marker='o',ms=MarkerSize)
        line3 = cur_ax.semilogy(t[2:],SamplingVar_empirical,color='C0',linewidth=lw,zorder=1)
        line4 = cur_ax.semilogy(t[2:],DriftVar_empirical,color='C2',linewidth=lw,zorder=1)
        cur_ax.set_xlim((0,T+9))
        cur_ax.set_ylim((ymin, ymax))
        
        if ns_idx == 0:
            cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
            if set_idx == 0:
                legend1 = cur_ax.legend(handles=[line1[0],line2[0]],labels=['Var$[\\^s]$',r'$\sigma_{\mathrm{s}}^2[\^s] + \sigma_{\mathrm{d}}^2[\^s]$'],fontsize=FontSize,frameon=False,loc=1,bbox_to_anchor=(0.65,1))
                cur_ax.add_artist(legend1)
                cur_ax.legend(handles=[line3[0],line4[0]],labels=[r'$\sigma_{\mathrm{s}}^2[\^s]$',r'$\sigma_{\mathrm{d}}^2[\^s]$'],fontsize=FontSize,frameon=False,loc=1)
        else:
            cur_ax.axes.get_yaxis().set_ticklabels([])
          
        if set_idx == 0:
            cur_ax.set_title('$n_s = $' + str(ns), fontsize=FontSize)  
        if set_idx == len(Sets)-1:
            cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
        else:
            cur_ax.axes.get_xaxis().set_ticklabels([])

        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
        cur_ax.text(0.05,0.09,'$N = $' + str(N),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)

plt.tight_layout()



if save_flag:
    plt.savefig('./Figures/FigR1_2.jpg',dpi=dpi_val,bbox_inches='tight')   