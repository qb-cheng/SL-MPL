# -*- coding: utf-8 -*-
"""
Plot the empirical and analytical estimator mean and variance, 
in the stochastic scenario under the joint effect of limited sampling and genetic drift.
Show that the mean of the MPL-based estimator is close to the true selection coefficient.
Show that the variance of the MPL-based estimator can be decomposed to sampling-only and drift-only variance,
thus allowing separate studies on these effects.
"""

# Plot the estimator mean and variance in the stochastic evolutionary scenario
# Mean and variance are "decomposed" into components due to sampling effects and due to drift effects respectively.
import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
from scipy.optimize import curve_fit

save_flag = True
dpi_val = 600
FontSize = 16
MarkerSize = 3
lw = 1.5

Palette = sns.color_palette('Set2')

thisSet = 11
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)

dt = 10
ns = 20
MCRun = 1000000

ymin = 1e-6
ymax = 1e-2

t = np.arange(0,T,dt)

with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
    StoTrajs = pickle.load(f)


# Compute the empirical estimator mean
# 1st col: under joint effects of finite sampling and genetic drift;
# 2nd col: under genetic drift only.
JointMean = np.zeros((len(t)-2,2))

# Compute the empirical estimator variance under joint effects of finite sampling and genetic drift
JointVar = np.zeros(len(t)-2)

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




fig,axes = plt.subplots(1,3,figsize=(15,4.5),dpi=dpi_val)

# Plot the empirical estimator mean under sampling and drift effects
cur_ax = axes[0]
cur_ax.plot(t[2:],JointMean[:,0],label='$\\mathrm{E}[\\^s]$',color=Palette[0],linewidth=lw) # (,linestyle='dashdot', $n_s = $'+str(ns)+')
cur_ax.plot(t[2:],JointMean[:,1],label='$\\mathrm{E}_{\\mathrm{d}}[\\^s_{\\mathrm{MPL}}]$',color=Palette[3],linewidth=lw,linestyle='--') # ,linestyle='dashdot'
cur_ax.axhline(y = s, color ="gray", linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$')

cur_ax.legend(fontsize=FontSize,frameon=False)
cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator mean',fontsize=FontSize)
cur_ax.set_xlim((0,T+9))
cur_ax.set_ylim((0.01,0.04))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))




# Plot the empirical estimator variance under sampling and drift effects
cur_ax = axes[1]
# Ts = np.arange(50,T,50)
# cur_ax.semilogy(Ts,JointVar[(Ts/dt).astype(int) - 2],label='Empirical Var$[\^s_{\mathrm{FS}}]$',color='C3',linewidth=lw,linestyle='None',marker='^',ms=MarkerSize,zorder=2)
cur_ax.semilogy(t[2:],JointVar,label='Var$[\\^s]$',color='C8',linewidth=lw) # ,linestyle='dashdot'
# cur_ax.semilogy(t[2:],SamplingVar_empirical,label=r'$\sigma_{\mathrm{s}}^2[\^s_{\mathrm{FS}}]$',color='C0',linewidth=lw,zorder=0,linestyle='dashdot')
# cur_ax.semilogy(t[2:],DriftVar_empirical,label=r'$\sigma_{\mathrm{d}}^2[\^s_{\mathrm{FS}}]$',color='C1',linewidth=lw,zorder=0,linestyle='dashdot')

SumVar = SamplingVar_empirical + DriftVar_empirical
InterestingLengths = np.arange(50,T,50) # np.array([50,100,150,300,450])
InterestingEstVar = np.zeros(len(InterestingLengths))
for T_idx,T_interest in enumerate(InterestingLengths):
    InterestingEstVar[T_idx] = SumVar[np.where(t == T_interest)[0][0] - 2]

# cur_ax.semilogy(t[2:],SumVar,label=r'$\sigma_{\mathrm{s}}^2[\^s_{\mathrm{FS}}] + \sigma_{\mathrm{d}}^2[\^s_{\mathrm{FS}}]$',color='C6',linestyle='None',marker='o',ms=MarkerSize)
cur_ax.semilogy(InterestingLengths,InterestingEstVar,label=r'$\sigma_{\mathrm{s}}^2[\^s] + \sigma_{\mathrm{d}}^2[\^s]$',color='c',linestyle='None',marker='o',ms=MarkerSize)

cur_ax.legend(fontsize=FontSize,frameon=False)
cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
cur_ax.set_xlim((0,T+9))
cur_ax.set_ylim((ymin, ymax))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))

# Plot the empirical and analytical sampling-only and drift-only variance to 
# show that our analytical approximation works
cur_ax = axes[2]
EmpiVar_V_log = np.log(SamplingVar_empirical)
popt_V = curve_fit(Functions.V_fit_model, np.log(DeterIV), EmpiVar_V_log, p0=EmpiVar_V_log[0])
line1 = cur_ax.semilogy(t[2:],SamplingVar_empirical,color='C0',linewidth=lw,zorder=1,label=r'$\sigma_{\mathrm{s}}^2[\^s]$') # ,linestyle='dashdot'
line2 = cur_ax.semilogy(t[2:],np.exp(popt_V[0])/(DeterIV**2),color=Palette[7],linewidth=lw,zorder=0,linestyle='--',label='Scaled $V^{-2}$')
line3 = cur_ax.semilogy(t[2:],DriftVar_empirical,color='C2',linewidth=lw,zorder=1,label=r'$\sigma_{\mathrm{d}}^2[\^s]$') # ,linestyle='dashdot'
line4 = cur_ax.semilogy(t[2:],CRLB,color='C3',linewidth=lw,zorder=2,linestyle='dotted',label='CRLB')
line5 = cur_ax.semilogy(t[2:],1 / (N * DeterIV),color='C4',linewidth=lw,zorder=0,linestyle='--',label='$N^{-1}V^{-1}$')
cur_ax.legend(fontsize=FontSize,frameon=False)
cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
cur_ax.set_xlim((0,T+9))
cur_ax.set_ylim((ymin, ymax))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y' ,labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))


axes[0].text(-0.18,1.15,'A',fontsize=18,transform=axes[0].transAxes,fontweight='bold',va='top',ha='right')
axes[1].text(-0.18,1.15,'B',fontsize=18,transform=axes[1].transAxes,fontweight='bold',va='top',ha='right')
axes[2].text(-0.18,1.15,'C',fontsize=18,transform=axes[2].transAxes,fontweight='bold',va='top',ha='right')

plt.tight_layout()



if save_flag:
    plt.savefig('./Figures/Fig5_EstMeanVarSto.pdf',dpi=dpi_val,bbox_inches='tight')