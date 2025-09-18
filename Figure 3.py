# -*- coding: utf-8 -*-
"""
Plot the empirical and analytical estimator mean and variance,
in the deterministic scenario under the effect of limited sampling.
Show that the mean is close to the true selection coefficient, while the variance follows the scaling we analyzed.
"""


import Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.optimize import curve_fit

save_flag = True

dpi_val = 600
FontSize = 16
MarkerSize = 4
lw = 1.5


dt = 10
MCRun = 1000000



thisSet = 11
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)


t = np.arange(0,T,dt)

InterestingLengths = np.arange(50,T,50) # np.array([50,100,150,300,450])

DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)
DeterIV = Functions.ComputeIV(DeterTraj[t], dt)


ns = 20

Palette = sns.color_palette('Set1')



fig,axes = plt.subplots(1,3,figsize=(15,4.5),dpi=dpi_val) 


# Plot estimator mean from observations of mutant allele frequency trajectory under deterministic evolutionary model
cur_ax = axes[0]
Sampledtrajs = Functions.SampleTrajectory(DeterTraj[t], ns, MCRun)
s_FS = Functions.SLMPL(Sampledtrajs, dt, ns, u)
EmpiMean = np.zeros(s_FS.shape[0])  
for t_idx in range(s_FS.shape[0]):
    cur_ests = s_FS[t_idx,:]
    cur_ests = cur_ests[np.isfinite(cur_ests)]
    EmpiMean[t_idx] = np.mean(cur_ests)


# Get analytical estimator mean from 2nd order multivariate Taylor series expansion
ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t], dt, ns, u)
TheoMean = ED/EV #- CovDV/EV**2 + (ED/EV)*VarV/EV**2
# cur_ax.plot(DeterIV[(InterestingLengths/dt).astype(int) - 2],TheoMean[(InterestingLengths/dt).astype(int) - 2],linestyle='None',marker='^',ms=MarkerSize,color=CustomizedColor[ns_idx])

cur_ax.plot(DeterIV,EmpiMean,label='E$[\\^s]$',color=Palette[4],linewidth=lw)
cur_ax.plot(DeterIV,TheoMean,color=Palette[3],linewidth=lw,label='$\\^s_{\\mathrm{MPL}}$',linestyle="--")

cur_ax.axhline(y = s, color ="gray", linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$')
cur_ax.legend(fontsize=FontSize,frameon=False,loc=1) #,bbox_to_anchor=(1, 0.85)
cur_ax.set_xlabel('Integrated variance, $V$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator mean',fontsize=FontSize)
cur_ax.set_xlim((0,np.max(DeterIV)+2))
cur_ax.set_ylim((0.01,0.04))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))







# Plot estimator variance with respect to sample size, and with respect to integrated variance
ns_vals = np.arange(10,201,10)

# Estimator variance in terms of ns
cur_ax = axes[1]

EmpiVar_ns = np.zeros(len(ns_vals))
AnalyticalVar_ns = np.zeros(len(ns_vals))
for ns_idx,cur_ns in enumerate(ns_vals):
    SampledTrajs = Functions.SampleTrajectory(DeterTraj[t],cur_ns,MCRun)
    
    D_hat = SampledTrajs[-1,:] - SampledTrajs[0,:] - u*dt*np.sum(1-2*SampledTrajs[:-1,:],axis=0)
    V_hat = dt * np.sum(SampledTrajs[:-1,:]*(1-SampledTrajs[:-1,:]),axis=0) / (1-1/cur_ns)
    sFS = D_hat / V_hat
    sFS = sFS[np.isfinite(sFS)]
    EmpiVar_ns[ns_idx] = np.var(sFS)
    
    ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t],dt,cur_ns,u)
    AnalyticalVar_ns[ns_idx] = (VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2)[-1]

EmpiVar_ns_log = np.log(EmpiVar_ns)
popt_ns = curve_fit(Functions.ns_fit_model, np.log(ns_vals), EmpiVar_ns_log, p0=EmpiVar_ns_log[0])


cur_ax.semilogy(ns_vals,EmpiVar_ns,linewidth=lw,color=Palette[1],label='Empirical Var[$\\^s$]')
cur_ax.semilogy(ns_vals,AnalyticalVar_ns,linewidth=lw,color=Palette[0],label='Analytical Var[$\\^s$]',linestyle='none',marker='o',ms=MarkerSize)
cur_ax.semilogy(ns_vals,np.exp(popt_ns[0])/ns_vals,linewidth=lw,linestyle='--',color='grey',label='Scaled $n_s^{-1}$',zorder=3)
cur_ax.set_xlim((0,np.max(ns_vals)+10))
cur_ax.set_ylim((1e-7,1e-4))
cur_ax.set_xlabel('Sample size, $n_s$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.legend(frameon=False,fontsize=FontSize,loc=1) #,bbox_to_anchor=(1, 0.9)




# Estimator variance in terms of V

cur_ax = axes[2]
SampledTrajs = Functions.SampleTrajectory(DeterTraj[t],ns,MCRun)
sFS_all = Functions.SLMPL(SampledTrajs,dt,ns,u)

EmpiVar_V = np.zeros(sFS_all.shape[0])
for t_idx,CurEst in enumerate(sFS_all):
    CurEst = CurEst[np.isfinite(CurEst)]
    EmpiVar_V[t_idx] = np.var(CurEst)

ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(DeterTraj[t],dt,ns,u)
AnalyticalVar_V = (VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2)

EmpiVar_V_log = np.log(EmpiVar_V)
popt_V = curve_fit(Functions.V_fit_model, np.log(DeterIV), EmpiVar_V_log, p0=EmpiVar_V_log[0])

InterestingEstVar = np.zeros(len(InterestingLengths))
InterestingV = np.zeros(len(InterestingLengths))
for T_idx,T_interest in enumerate(InterestingLengths):
    InterestingV[T_idx] = DeterIV[np.where(t == T_interest)[0][0] - 2]
    InterestingEstVar[T_idx] = AnalyticalVar_V[np.where(t == T_interest)[0][0] - 2]
    # InterestingEstVar[T_idx] = EmpiVar_V[np.where(t == T_interest)[0][0] - 2]

cur_ax.semilogy(DeterIV,EmpiVar_V,linewidth=lw,color=Palette[1],label='Empirical Var[$\\^s$]')
cur_ax.semilogy(InterestingV,InterestingEstVar,linewidth=lw,color=Palette[0],label='Analytical Var[$\\^s$]',linestyle='none',marker='o',ms=MarkerSize)
cur_ax.semilogy(DeterIV,np.exp(popt_V[0])/(DeterIV**2),linewidth=lw,linestyle='--',color='grey',label='Scaled $V^{-2}$',zorder=3)
cur_ax.set_xlim((0,np.max(DeterIV)+2))
cur_ax.set_ylim((1e-6,1e-2))
cur_ax.set_xlabel('Integrated variance, $V$',fontsize=FontSize)
cur_ax.set_ylabel('',fontsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.legend(frameon=False,fontsize=FontSize,loc=1) #,bbox_to_anchor=(1,0.9)



axes[0].text(-0.17,1.15,'A',fontsize=18,transform=axes[0].transAxes,fontweight='bold',va='top',ha='right')
axes[1].text(-0.175,1.15,'B',fontsize=18,transform=axes[1].transAxes,fontweight='bold',va='top',ha='right')



plt.tight_layout()

cur_ax = axes[1]
x_pos = cur_ax.get_position().intervalx
y_pos = cur_ax.get_position().intervaly
cur_ax.set_position([x_pos[0]+0.02, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])


if save_flag:
    plt.savefig('./Figures/Fig3_EstMeanVarDeter.pdf',dpi=dpi_val,bbox_inches='tight')