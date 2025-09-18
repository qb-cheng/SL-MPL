# -*- coding: utf-8 -*-
"""
Study the estimator performance (mean and variance) with respect to the time sampling step delta t.
Given enough informative samples (i.e., away from the boundary) from the observed mutant allele frequency trajectories with fixed length T,
plot the sampling-only and drift-only variance.
"""



import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
import pandas as pd


save_flag = True

dpi_val = 600
FontSize = 16
MarkerSize = 5
lw = 1.5
Palette = sns.color_palette('Set2')

ns = 20
MCRun = 100000

thisSet = 11
xstep = 10

N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
T = 151
# dts = np.array([1,2,3,5,6,10,15,25,30,50])
dts = np.array([factor for factor in range (1,T) if (T-1) % factor == 0 and factor <= (T-1)/3])

# Quasispecies trajectory taken as reference to decide maximum dt for each case
# DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)
# T_end = np.where(DeterTraj <= 0.9)[0][-1]
# dt_max = T_end / 2

# dts = dts[dts < dt_max]


# Stochastic evolutionary scenario
with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
    StoTrajs = pickle.load(f)
    StoTrajs = StoTrajs[:T]
    
SampledStoTrajs = Functions.SampleOnce(StoTrajs, ns)  
    
MeanTraj = np.mean(StoTrajs,axis=1)
SampledMeanTrajs = Functions.SampleTrajectory(MeanTraj, ns, MCRun)


# Get empirical estimator mean
EmpiMean_joint = np.zeros(len(dts))

# Get empirical estimator variances
EmpiVar_joint = np.zeros(len(dts))
EmpiVar_sampling = np.zeros(len(dts))
AnalyVar_sampling = np.zeros(len(dts)) # Computed from the mean trajectory, not the deterministic trajectory
EmpiVar_drift = np.zeros(len(dts))
CRLB_drift = np.zeros(len(dts)) # CRLB of the drift-only variance

df_est = pd.DataFrame()
estimates = []
SamplingSteps = []
dt_for_distribution_plot = [1,5,10,25,50]

for dt_idx,dt in enumerate(dts):
    t = np.arange(0,T,dt)
    
    # The joint variance w.r.t. dt in the stochastic evolutionary scenario
    ObservedSampledStoTrajs = SampledStoTrajs[t]
    D_hat = ObservedSampledStoTrajs[-1,:] - ObservedSampledStoTrajs[0,:] - u*dt*np.sum(1-2*ObservedSampledStoTrajs[:-1,:],axis=0)
    V_hat = dt * np.sum(ObservedSampledStoTrajs[:-1,:]*(1-ObservedSampledStoTrajs[:-1,:]),axis=0) / (1-1/ns)
    s_FS = D_hat / V_hat
    s_FS = s_FS[np.isfinite(s_FS)]
    EmpiMean_joint[dt_idx] = np.mean(s_FS)
    EmpiVar_joint[dt_idx] = np.var(s_FS)
    
    if dt in dt_for_distribution_plot:
        estimates += list(s_FS)
        SamplingSteps += [dt]*len(s_FS)
    
    # The sampling-only variance w.r.t. dt in the stochastic evolutionary scenario
    ObservedSampledMeanTrajs = SampledMeanTrajs[t]
    D_hat = ObservedSampledMeanTrajs[-1,:] - ObservedSampledMeanTrajs[0,:] - u*dt*np.sum(1-2*ObservedSampledMeanTrajs[:-1,:],axis=0)
    V_hat = dt * np.sum(ObservedSampledMeanTrajs[:-1,:]*(1-ObservedSampledMeanTrajs[:-1,:]),axis=0) / (1-1/ns)
    s_FS = D_hat / V_hat
    s_FS = s_FS[np.isfinite(s_FS)]
    EmpiVar_sampling[dt_idx] = np.var(s_FS)
    
    ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(MeanTraj[t], dt, ns, u)
    AnalyVar_sampling[dt_idx] = (VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2)[-1]
    
    # The drift-only variance w.r.t. dt in the stochastic evolutionary scenario
    ObservedStoTrajs = StoTrajs[t]
    D = ObservedStoTrajs[-1,:] - ObservedStoTrajs[0,:] - u*dt*np.sum(1-2*ObservedStoTrajs[:-1,:],axis=0)
    V = dt * np.sum(ObservedStoTrajs[:-1,:]*(1-ObservedStoTrajs[:-1,:]),axis=0)
    s_hat = D / V
    EmpiVar_drift[dt_idx] = np.var(s_hat)
    MeanIV = np.mean(np.sum(dt * ObservedStoTrajs[:-1] * (1-ObservedStoTrajs[:-1]), axis=0))
    CRLB_drift[dt_idx] = 1 / (N*MeanIV)
    
df_est['estimates'] = np.array(estimates)
df_est['SamplingSteps'] = SamplingSteps

ymin = 1e-6
ymax = 1e-2
var_max = 1e-4
# var_max = np.max([np.max(EmpiVar_sampling),np.max(EmpiVar_drift)])
# var_max_log = np.floor(np.log10(var_max))
# if var_max / (10**var_max_log) <= 5:
#     var_max = 5 * (10**var_max_log)
# else:
#     var_max = 10 ** (var_max_log+1)


fig,axes = plt.subplots(2,2,figsize=(11,8),dpi=dpi_val) 

# Subplot 1: examples of observed frequency trajectories under the joint effects of drift and sampling
cur_ax = axes[0,0]
# cur_ax.plot(np.arange(T),SampledStoTrajs[:,0],color='gray',linewidth=0.5,label='Observations\n($N = $'+str(N)+', $n_s = $'+str(ns)+')',zorder=1)
# cur_ax.plot(np.arange(T),DeterTraj,color='b',linewidth=lw,label='Quasispecies',zorder=2)
cur_ax.plot(np.arange(T),SampledStoTrajs[:,:5],linewidth=0.75,alpha=0.5,zorder=0)
# cur_ax.legend(fontsize=FontSize,frameon=False)
cur_ax.set_xlabel('Generation, $t$',fontsize=FontSize)
cur_ax.set_ylabel('Mutant allele frequency',fontsize=FontSize)
cur_ax.set_xlim((0,T+3))
cur_ax.set_ylim((-0.05,1.05))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  

# Subplot 2: show the empirical distribution of estimates with the smallest and the largest dt
cur_ax = axes[0,1]
estimates_smallest_dt = df_est['estimates'][df_est['SamplingSteps'] == np.min(dts)]
estimates_largest_dt = df_est['estimates'][df_est['SamplingSteps'] == np.max(dts)]
xmin = -0.04
xmax = 0.08
bins = np.linspace(xmin,xmax,50)
cur_ax.hist(estimates_smallest_dt,bins=bins,alpha=0.4,density=True,edgecolor='black',linewidth=lw/2,label='$\Delta t = $'+str(np.min(dts)))
cur_ax.hist(estimates_largest_dt,bins=bins,alpha=0.4,density=True,edgecolor='black',linewidth=lw/2,label='$\Delta t = $'+str(np.max(dts)))
cur_ax.axvline(x = s, color ="gray", linestyle ="dotted",linewidth=lw)
cur_ax.set_xlim((xmin,xmax))
cur_ax.set_ylim((0,60))
cur_ax.set_xlabel('Selection coefficient estimate, $\\^s$',fontsize=FontSize)
cur_ax.set_ylabel('Density',fontsize=FontSize)
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.04))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)


# Subplot 3: show that the joint mean is close to the true selection coefficient for various dt
cur_ax = axes[1,0]
cur_ax.plot(dts,EmpiMean_joint,label='$\\mathrm{E}[\\^s]$',color=Palette[0],linestyle='None',marker='o',ms=MarkerSize,markeredgewidth=2,markerfacecolor='w',zorder=2) # (,linestyle='dashdot', $n_s = $'+str(ns)+')
cur_ax.axhline(y = s, color ="gray", linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$',zorder=1)
cur_ax.legend(fontsize=FontSize,frameon=False)
cur_ax.set_xlabel('Time sampling step, $\Delta t$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator mean',fontsize=FontSize)
cur_ax.set_xlim((0,np.max(dts) + 2))
if s == 0.01:
    cur_ax.set_ylim((0,0.03))
elif s == 0.05:
    cur_ax.set_ylim((0.04,0.07))
else:
    cur_ax.set_ylim((0.01,0.04))
# cur_ax.set_ylim((0,0.08))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))


# Subplot 4: show that the joint variance can still be decomposed into the sum of sampling-only and drift-only variance
cur_ax = axes[1,1]
cur_ax.semilogy(dts,EmpiVar_joint,label='Var$[\\^s]$',color='c',linestyle='None',marker='o',ms=MarkerSize,markeredgewidth=2,markerfacecolor='w')
cur_ax.set_ylim((ymin, ymax))
cur_ax.legend(fontsize=FontSize,frameon=False,loc=1)

# cur_ax.plot(dts,EmpiVar_joint,label='Var$[\\^s]$',color='C8',linewidth=lw)
# cur_ax.plot(dts,SumVar,label=r'$\sigma_{\mathrm{s}}^2[\^s] + \sigma_{\mathrm{d}}^2[\^s]$',color='c',linestyle='None',marker='o',ms=MarkerSize)
# cur_ax.set_ylim((0, var_max))
# cur_ax.legend(fontsize=FontSize,frameon=False,loc=4)

cur_ax.set_xlabel('Time sampling step, $\Delta t$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
cur_ax.set_xlim((0,np.max(dts) + 2))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))




plt.tight_layout()


cur_ax = axes[0,0]
x_pos = cur_ax.get_position().intervalx
y_pos = cur_ax.get_position().intervaly
cur_ax.set_position([x_pos[0]-0.03, y_pos[0]+0.02, x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
fig.text(0,1.03,'A',fontsize=18,fontweight='bold',va='top',ha='right')

cur_ax = axes[0,1]
x_pos = cur_ax.get_position().intervalx
y_pos = cur_ax.get_position().intervaly
cur_ax.set_position([x_pos[0]+0.03, y_pos[0]+0.02, x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
fig.text(0.54,1.03,'B',fontsize=18,fontweight='bold',va='top',ha='right')

cur_ax = axes[1,0]
x_pos = cur_ax.get_position().intervalx
y_pos = cur_ax.get_position().intervaly
cur_ax.set_position([x_pos[0]-0.03, y_pos[0]-0.02, x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
fig.text(0,0.5,'C',fontsize=18,fontweight='bold',va='top',ha='right')

cur_ax = axes[1,1]
x_pos = cur_ax.get_position().intervalx
y_pos = cur_ax.get_position().intervaly
cur_ax.set_position([x_pos[0]+0.03, y_pos[0]-0.02, x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
fig.text(0.54,0.5,'D',fontsize=18,fontweight='bold',va='top',ha='right')




if save_flag:
    plt.savefig('./Figures/Fig6_dtEffects.pdf',dpi=dpi_val,bbox_inches='tight')
        