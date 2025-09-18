# -*- coding: utf-8 -*-
"""
Empirical and analytical estimator mean and variance,
under different values of sample sizes and initial mutant allele frequencies.
This is to show that our analytical results match empirical observations.
"""

import Functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.optimize import curve_fit

save_flag = True

dpi_val = 600

FigSizeR = 4
FontSize = 14
MarkerSize = 4
lw = 1.5

u = 1e-3
dt = 10
MCRun = 1000000

s_ref = 0.02
# x0_ref = 0.1

T = 451
u = 1e-3

nss = [10,20,50]
palette_indices = [6,4,2]
ns_vals = np.arange(10,201,10)
ns_ref = 20

InterestingLengths = np.arange(50,T,50)


Palette = sns.color_palette('Set1')

x0s = [0.1,0.3,0.5]
# x0s = [0.1,0.2,0.3,0.4,0.5]


fig,axes = plt.subplots(len(x0s),4,figsize=(15,9),dpi=dpi_val) 




for x0_idx,cur_x0 in enumerate(x0s):
    t = np.arange(0,T,dt)
    
    # First column: Population mutant allele frequency trajectory in the deterministic scenario
    DeterTraj = Functions.DeterministicTrajectory(s_ref,cur_x0,T,u)
    
    cur_ax = axes[x0_idx,0]
    cur_ax.plot(t,DeterTraj[t],color='gray',linewidth=lw)
    cur_ax.set_xlim((0,T+9))
    if x0_idx == len(x0s)-1:
        cur_ax.set_xlabel('Generation, $t$',fontsize=FontSize)
    else:
        cur_ax.set_xlabel('',fontsize=FontSize)
        cur_ax.axes.get_xaxis().set_ticklabels([])
    cur_ax.set_ylim((0,1))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
    cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    cur_ax.text(0.95,0.1,'$x(0) = $'+str(cur_x0),fontsize=FontSize,ha='right',va='center',transform=cur_ax.transAxes)
    cur_ax.set_ylabel('Mutant allele frequency, $x$',fontsize=FontSize)

    
    DeterIV = Functions.ComputeIV(DeterTraj[t], dt)
    
    """
    # Second row: Integrated variance
    cur_ax = axes[1,x0_idx]
    cur_ax.plot(t[2:],DeterIV,color='C4',linewidth=lw)
    cur_ax.set_xlim((0,T+9))
    cur_ax.set_xlabel('Generations used for inference, $T$',fontsize=FontSize)
    cur_ax.set_ylim((0,60))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
    cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
    if x0_idx == 0: 
        cur_ax.set_ylabel('Integrated variance, $V$',fontsize=FontSize)
    else:
        cur_ax.set(ylabel=None)
        plt.setp(cur_ax.get_yticklabels(), visible=False)
    """
    
    
    
    
    # Second column: Analytical and empirical estimator mean
    cur_ax = axes[x0_idx,1]
    for ns_idx,cur_ns in enumerate(nss):  
        # Get analytical estimator mean from 2nd order multivariate Taylor series expansion
        ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t], dt, cur_ns, u)
        # TheoMean = ED/EV - CovDV/EV**2 + (ED/EV)*VarV/EV**2
        # cur_ax.plot(DeterIV,TheoMean,color=Palette[ns_idx],linewidth=lw,label='$n_s$ = '+str(cur_ns))
        TheoMean = ED/EV

        # Get empirical estimator mean from observations of mutant allele frequency trajectory under deterministic evolutionary model
        SampledTrajs = Functions.SampleTrajectory(DeterTraj[t], cur_ns, MCRun)
        s_FS = Functions.SLMPL(SampledTrajs, dt, cur_ns, u)
        EmpiMean = np.zeros(s_FS.shape[0])  
        for t_idx in range(s_FS.shape[0]):
            cur_ests = s_FS[t_idx,:]
            cur_ests = cur_ests[np.isfinite(cur_ests)]
            EmpiMean[t_idx] = np.mean(cur_ests)
        # cur_ax.plot(DeterIV[(InterestingLengths/dt).astype(int) - 2],EmpiMean[(InterestingLengths/dt).astype(int) - 2],linestyle='None',marker='^',ms=MarkerSize,color=Palette[ns_idx])
        cur_ax.plot(DeterIV,EmpiMean,color=Palette[palette_indices[ns_idx]],linewidth=lw,label='$n_s$ = '+str(cur_ns))
    cur_ax.plot(DeterIV,TheoMean,color=Palette[3],linewidth=lw,label='$\\^s_{\\mathrm{MPL}}$',linestyle="--")

    if x0_idx == len(x0s)-1:
        cur_ax.set_xlabel('Integrated variance, $V$',fontsize=FontSize)
    else:
        cur_ax.set_xlabel('',fontsize=FontSize)
        cur_ax.axes.get_xaxis().set_ticklabels([])
    # cur_ax.set_xlim((0,np.max(DeterIV)+2))
    cur_ax.set_xlim((0,60))
    cur_ax.set_ylim((0.01,0.05))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    cur_ax.axhline(y = s_ref, color ="gray", linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$')
    cur_ax.set_ylabel('Estimator mean, E$[\\^s]$',fontsize=FontSize)
    cur_ax.text(0.95,0.1,'$x(0) = $'+str(cur_x0),fontsize=FontSize,ha='right',va='center',transform=cur_ax.transAxes)

    if x0_idx == 0:
        cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)
    
    
    
    
    
    
    # Fourth column: Analytical and empirical estimator variance, as a function of integrated variance V
    cur_ax = axes[x0_idx,3]
    
    ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t],dt,ns_ref,u)
    AnalyticalVar_V = (VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2)
    
    SampledTraj = Functions.SampleTrajectory(DeterTraj[t],ns_ref,MCRun)
    AllEsts = Functions.SLMPL(SampledTraj,dt,ns_ref,u,)
    EmpiVar_V = np.zeros(AllEsts.shape[0])
    for t_idx,CurEst in enumerate(AllEsts):
        CurEst = CurEst[np.isfinite(CurEst)]
        EmpiVar_V[t_idx] = np.var(CurEst)

    EmpiVar_V_log = np.log(EmpiVar_V)
    popt_V = curve_fit(Functions.V_fit_model, np.log(DeterIV), EmpiVar_V_log, p0=EmpiVar_V_log[0])

    InterestingEstVar = np.zeros(len(InterestingLengths))
    InterestingV = np.zeros(len(InterestingLengths))
    for T_idx,T_interest in enumerate(InterestingLengths):
        InterestingV[T_idx] = DeterIV[np.where(t == T_interest)[0][0] - 2]
        InterestingEstVar[T_idx] = AnalyticalVar_V[np.where(t == T_interest)[0][0] - 2]

    cur_ax.semilogy(DeterIV,EmpiVar_V,linewidth=lw,color=Palette[1],label='Empirical')
    cur_ax.semilogy(InterestingV,InterestingEstVar,linewidth=lw,color=Palette[0],label='Analytical',linestyle='none',marker='o',ms=MarkerSize)
    cur_ax.semilogy(DeterIV,np.exp(popt_V[0])/(DeterIV**2),linewidth=lw,linestyle='--',color='grey',label='Scaled $V^{-2}$',zorder=3)
    cur_ax.set_xlim((0,60))
    cur_ax.set_ylim((1e-6,1e-2))
    if x0_idx == len(x0s)-1:
        cur_ax.set_xlabel('Integrated variance, $V$',fontsize=FontSize)
    else:
        cur_ax.set_xlabel('',fontsize=FontSize)
        cur_ax.axes.get_xaxis().set_ticklabels([])
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.set_ylabel('Estimator variance, Var[$\\^s$]',fontsize=FontSize)
    cur_ax.text(0.05,0.1,'$x(0) = $'+str(cur_x0),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)

    if x0_idx == 0:
        cur_ax.legend(frameon=False,fontsize=FontSize,loc=1)
    


    # Third column: Analytical and empirical estimator variance, as a function of sample size ns
    cur_ax = axes[x0_idx,2]
    EmpiVar_ns = np.zeros(len(ns_vals))
    AnalyticalVar_ns = np.zeros(len(ns_vals))
    for ns_idx,cur_ns in enumerate(ns_vals):
        ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t],dt,cur_ns,u)
        AnalyticalVar_ns[ns_idx] = (VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2)[-1]
        
        SampledTraj = Functions.SampleTrajectory(DeterTraj[t],cur_ns,MCRun)
        ests = Functions.SLMPL(SampledTraj,dt,cur_ns,u)[-1,:]
        ests = ests[np.isfinite(ests)]
        EmpiVar_ns[ns_idx] = np.var(ests)
        
    EmpiVar_ns_log = np.log(EmpiVar_ns)
    popt_ns = curve_fit(Functions.ns_fit_model, np.log(ns_vals), EmpiVar_ns_log, p0=EmpiVar_ns_log[0])
    
    cur_ax.semilogy(ns_vals,EmpiVar_ns,linewidth=lw,color=Palette[1],label='Empirical')
    cur_ax.semilogy(ns_vals,AnalyticalVar_ns,linewidth=lw,color=Palette[0],label='Analytical',linestyle='none',marker='o',ms=MarkerSize)
    cur_ax.semilogy(ns_vals,np.exp(popt_ns[0])/ns_vals,linewidth=lw,linestyle='--',color='grey',label='Scaled $n_s^{-1}$',zorder=3)
    cur_ax.set_xlim((0,np.max(ns_vals)+10))
    cur_ax.set_ylim((1e-7,1e-4))
    if x0_idx == len(x0s)-1:
        cur_ax.set_xlabel('Sample size, $n_s$',fontsize=FontSize)
    else:
        cur_ax.set_xlabel('',fontsize=FontSize)
        cur_ax.axes.get_xaxis().set_ticklabels([])
    cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.set_ylabel('Estimator variance, Var[$\\^s$]',fontsize=FontSize)
    cur_ax.text(0.05,0.1,'$x(0) = $'+str(cur_x0),fontsize=FontSize,ha='left',va='center',transform=cur_ax.transAxes)

    if x0_idx == 0:
        cur_ax.legend(frameon=False,fontsize=FontSize,loc=1)


axes[0,0].text(-0.18,1.18,'A',fontsize=18,transform=axes[0,0].transAxes,fontweight='bold',va='top',ha='right')
axes[0,1].text(-0.22,1.18,'B',fontsize=18,transform=axes[0,1].transAxes,fontweight='bold',va='top',ha='right')
axes[0,2].text(-0.23,1.18,'C',fontsize=18,transform=axes[0,2].transAxes,fontweight='bold',va='top',ha='right')
axes[0,3].text(-0.23,1.18,'D',fontsize=18,transform=axes[0,3].transAxes,fontweight='bold',va='top',ha='right')


plt.tight_layout()



if save_flag:
    plt.savefig('./Figures/SuppFig2_Various_x0.pdf',dpi=dpi_val,bbox_inches='tight')


