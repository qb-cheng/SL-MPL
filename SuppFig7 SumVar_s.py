# -*- coding: utf-8 -*-
"""
Estimator mean and variance with different values of selection coefficients,
in the stochastic scenario under the joint effect of limited sampling and genetic drift.
Show that the estimator variance can be decomposed into the sum of sampling-only and drift-only variances.
"""

import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
from scipy.optimize import curve_fit

save_flag = True
dpi_val = 600
FontSize = 14
MarkerSize = 4
lw = 1.5

Palette = sns.color_palette('Set2')


dt = 10
ns = 20
MCRun = 1000000

ymin = 1e-6
ymax = 1e-2

# Sets = [13,10,11,12]
Sets = [14,13,10,12]

fig,axes = plt.subplots(len(Sets),3,figsize=(15,len(Sets)*3),dpi=dpi_val)
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
    t = np.arange(0,T,dt)
    
    with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
        StoTrajs = pickle.load(f)
        
    # Compute the empirical estimator variance under joint effects of finite sampling and genetic drift
    JointVar = np.zeros(len(t)-2)

    # Compute the empirical estimator mean under joint effects of finite sampling and genetic drift
    JointMean = np.zeros((len(t)-2,2))

    SampledStoTrajs = Functions.SampleOnce(StoTrajs[t], ns)
    sFS_joint = Functions.SLMPL(SampledStoTrajs, dt, ns, u)

    for idx in range(sFS_joint.shape[0]):
        CurEsts = sFS_joint[idx,:]
        CurEsts = CurEsts[np.isfinite(CurEsts)]
        JointMean[idx,0] = np.mean(CurEsts)
        JointVar[idx] = np.var(CurEsts)
        
    shat = Functions.SLMPL(StoTrajs[t,:], dt, 0, u)
    JointMean[:,1] = np.mean(shat,axis=1)


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
    
    # Compute the empirical CRLB
    StoIVs = np.zeros((len(t)-2,StoTrajs.shape[1]))
    for ItrIdx in range(StoTrajs.shape[1]):
        CurStoTraj = StoTrajs[:,ItrIdx]
        StoIVs[:,ItrIdx] = Functions.ComputeIV(CurStoTraj[t], dt)   
    StoIVs_mean = np.mean(StoIVs,axis=1)
    CRLB = 1 / (N*(StoIVs_mean))

    #Compute the analytical approximate closed-form sampling-only and drift-only variance
    DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)
    DeterIV = Functions.ComputeIV(DeterTraj[t], dt)
    ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t],dt,ns,u)
    SamplingVar_analytical = (VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2)
    DriftVar_analytical = 1 / (N * DeterIV);
    
    
    # Plot the empirical estimator mean under sampling and drift effects
    cur_ax = axes[set_idx,0]
    cur_ax.plot(t[2:],JointMean[:,0],label='$\\mathrm{E}[\\^s]$',color=Palette[0],linewidth=lw) # (,linestyle='dashdot', $n_s = $'+str(ns)+')
    cur_ax.plot(t[2:],JointMean[:,1],label='$\\mathrm{E}_{\\mathrm{d}}[\\^s_{\\mathrm{MPL}}]$ ',color=Palette[3],linewidth=lw,linestyle='--') # ,linestyle='dashdot'
    cur_ax.axhline(y = s, color ="gray", linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$')
    if set_idx == 0:
        cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)
    if set_idx == len(Sets)-1:
        cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
    else:
        cur_ax.axes.get_xaxis().set_ticklabels([])
    cur_ax.set_ylabel('Estimator mean',fontsize=FontSize)
    cur_ax.set_xlim((0,T+9))
    cur_ax.set_ylim((-0.01,0.08))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
    cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    cur_ax.text(0.05,0.91,'$s = $' + str(s),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)

    # Plot the empirical estimator variance under sampling and drift effects
    cur_ax = axes[set_idx,1]
    # cur_ax.semilogy(t[2:],JointVar,label='Var$[\^s_{\mathrm{FS}}]$',color='C1',linewidth=lw,zorder=2)
    
    cur_ax.semilogy(t[2:],JointVar,label='Var$[\\^s]$',color='C8',linewidth=lw,zorder=2)
    SumVar = SamplingVar_empirical + DriftVar_empirical
    InterestingLengths = np.arange(50,T,50) # np.array([50,100,150,300,450])
    InterestingEstVar = np.zeros(len(InterestingLengths))
    for T_idx,T_interest in enumerate(InterestingLengths):
        InterestingEstVar[T_idx] = SumVar[np.where(t == T_interest)[0][0] - 2]
    cur_ax.semilogy(InterestingLengths,InterestingEstVar,label=r'$\sigma_{\mathrm{s}}^2[\^s] + \sigma_{\mathrm{d}}^2[\^s]$',color='c',linestyle='None',marker='o',ms=MarkerSize)
    
    # cur_ax.semilogy(t[2:],SamplingVar_empirical + DriftVar_empirical,label=r'$\sigma_{\mathrm{s}}^2[\^s_{\mathrm{FS}}] + \sigma_{\mathrm{d}}^2[\^s_{\mathrm{FS}}]$',color='C9',linestyle='None',marker='o',ms=MarkerSize)
    if set_idx == 0:
        cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)
    if set_idx == len(Sets)-1:
        cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
    else:
        cur_ax.axes.get_xaxis().set_ticklabels([])
    cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
    cur_ax.set_xlim((0,T+9))
    cur_ax.set_ylim((ymin, ymax))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
    cur_ax.text(0.05,0.09,'$s = $' + str(s),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)

    # Plot the empirical and analytical sampling-only and drift-only variance to 
    # show that our analytical approximation works
    cur_ax = axes[set_idx,2]
    EmpiVar_V_log = np.log(SamplingVar_empirical)
    popt_V = curve_fit(Functions.V_fit_model, np.log(DeterIV), EmpiVar_V_log, p0=EmpiVar_V_log[0])
    line1 = cur_ax.semilogy(t[2:],SamplingVar_empirical,color='C0',linewidth=lw,zorder=1)
    line2 = cur_ax.semilogy(t[2:],np.exp(popt_V[0])/(DeterIV**2),color=Palette[7],linewidth=lw,zorder=0,linestyle='--')
    line3 = cur_ax.semilogy(t[2:],DriftVar_empirical,color='C2',linewidth=lw,zorder=1)
    line4 = cur_ax.semilogy(t[2:],CRLB,color='C3',linewidth=lw,zorder=2,linestyle='dotted')
    line5 = cur_ax.semilogy(t[2:],1 / (N * DeterIV),color='C4',linewidth=lw,zorder=0,linestyle='--')
    if set_idx == 0:
        legend1 = cur_ax.legend(handles=[line1[0],line2[0]],labels=[r'$\sigma_{\mathrm{s}}^2[\^s]$','Scaled $V^{-2}$'],fontsize=FontSize,frameon=False,loc=1,bbox_to_anchor=(0.6,1))
        cur_ax.add_artist(legend1)
        cur_ax.legend(handles=[line3[0],line4[0],line5[0]],labels=[r'$\sigma_{\mathrm{d}}^2[\^s]$','CRLB','$N^{-1}V^{-1}$'],fontsize=FontSize,frameon=False,loc=1)
    if set_idx == len(Sets)-1:
        cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
    else:
        cur_ax.axes.get_xaxis().set_ticklabels([])
    cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
    cur_ax.set_xlim((0,T+9))
    cur_ax.set_ylim((ymin, ymax))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
    cur_ax.text(0.05,0.09,'$s = $' + str(s),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)


axes[0,0].text(-0.16,1.15,'A',fontsize=18,transform=axes[0,0].transAxes,fontweight='bold',va='top',ha='right')
axes[0,1].text(-0.16,1.15,'B',fontsize=18,transform=axes[0,1].transAxes,fontweight='bold',va='top',ha='right')
axes[0,2].text(-0.16,1.15,'C',fontsize=18,transform=axes[0,-1].transAxes,fontweight='bold',va='top',ha='right')


plt.tight_layout()



if save_flag:
    plt.savefig('./Figures/SuppFig7_EstMeanVarSto_s.pdf',dpi=dpi_val,bbox_inches='tight')