# -*- coding: utf-8 -*-
"""
Estimator mean and variance with time-varying sample sizes,
where the sample size at each time point follows a Poisson distribution.
"""

import Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
 

save_flag = True
dpi_val = 350
FontSize = 14
MarkerSize = 2
lw = 1.5

xmin = -0.02
xmax = 0.06

thisSet = 11
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)

dt = 10
t = np.arange(0,T,dt)

lambda_nss = np.arange(10,201,10)
ns_interest = 20
MCRun = 1000000

Palette = sns.color_palette('Set1')

DeterTraj = Functions.DeterministicTrajectory(s, x0, T, u)
DeterIV = Functions.ComputeIV(DeterTraj[t], dt)

# Results obtained from estimators derived with (1) time-varying sample size, and (2) constant sample size
s_hat_TimeVarying = np.zeros((len(t)-2,MCRun,len(lambda_nss)))

for ns_idx, lambda_ns in enumerate(lambda_nss):
    for MCidx in range(MCRun):
        cur_ns = np.random.poisson(lambda_ns, len(t))
        while any(cur_ns < 2):
            cur_ns = np.random.poisson(lambda_ns, len(t))
            
        cur_ObTraj = np.zeros(len(t))
        for t_idx in range(len(t)):
            cur_ObTraj[t_idx] = np.random.binomial(cur_ns[t_idx], DeterTraj[t[t_idx]]) / cur_ns[t_idx]
            
        Dhat = cur_ObTraj[1:] - cur_ObTraj[0] - u*np.cumsum(dt*(1-2*cur_ObTraj[:-1]))
        Vhat_corrected = np.cumsum(dt * cur_ObTraj[:-1] * (1 - cur_ObTraj[:-1]) * cur_ns[:-1] / (cur_ns[:-1]-1))
            
        s_FS = Dhat / Vhat_corrected
        s_hat_TimeVarying[:,MCidx,ns_idx] = s_FS[1:]
        
        
        
    

est_mean_TimeVarying = np.zeros((len(t)-2,len(lambda_nss)))
est_var_TimeVarying = np.zeros((len(t)-2,len(lambda_nss)))
for ns_idx in range(len(lambda_nss)):
    for t_idx in range(len(t)-2):
        cur_est = s_hat_TimeVarying[t_idx,:,ns_idx]
        cur_est = cur_est[np.isfinite(cur_est)]
        est_mean_TimeVarying[t_idx,ns_idx] = np.mean(cur_est)
        est_var_TimeVarying[t_idx,ns_idx] = np.var(cur_est)
        
        
est_MPL = Functions.SLMPL(DeterTraj[t],dt,0,u)





fig,axes = plt.subplots(1,3,figsize=(15,4),dpi=dpi_val) 


# Plot estimator mean from observations of mutant allele frequency trajectory under deterministic evolutionary model
cur_ax = axes[0]
cur_ax.plot(DeterIV,est_mean_TimeVarying[:,np.where(lambda_nss == ns_interest)[0][0]],label='E$[\\^s]$',color=Palette[4],linewidth=lw)
cur_ax.plot(DeterIV,est_MPL,label='$\\^s_{\\mathrm{MPL}}$',color=Palette[3],linewidth=lw,linestyle="--")
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
    
# Estimator variance in terms of ns
cur_ax = axes[1]
ref_ns = est_var_TimeVarying[-1,0] * lambda_nss[0] / lambda_nss
cur_ax.semilogy(lambda_nss,est_var_TimeVarying[-1],linewidth=lw,color=Palette[1],label='Var$[\\^s]$',zorder=1)
cur_ax.semilogy(lambda_nss,ref_ns,linewidth=lw,linestyle='--',color='grey',label='Scaled $\overline{n}_s^{-1}$',zorder=2)
cur_ax.set_xlim((0,np.max(lambda_nss)+10))
cur_ax.set_ylim((1e-7,1e-4))
cur_ax.set_xlabel('Mean sample size, $\overline{n}_s$',fontsize=FontSize)
cur_ax.set_ylabel('Estimator variance',fontsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.legend(frameon=False,fontsize=FontSize,loc=1) #,bbox_to_anchor=(1, 0.9)

# Estimator variance in terms of V
cur_ax = axes[2]
ref_IV = est_var_TimeVarying[3,np.where(lambda_nss == ns_interest)[0][0]] * DeterIV[3]**2 / (DeterIV**2)
cur_ax.semilogy(DeterIV,est_var_TimeVarying[:,np.where(lambda_nss == ns_interest)[0][0]],linewidth=lw,color=Palette[1],label='Var$[\\^s]$',zorder=1)
cur_ax.semilogy(DeterIV,ref_IV,linewidth=lw,linestyle='--',color='grey',label='Scaled $V^{-2}$',zorder=2)
cur_ax.set_xlim((0,np.max(DeterIV)+2))
cur_ax.set_ylim((1e-6,1e-2))
cur_ax.set_xlabel('Integrated variance, $V$',fontsize=FontSize)
cur_ax.set_ylabel('',fontsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.legend(frameon=False,fontsize=FontSize,loc=1) #,bbox_to_anchor=(1,0.9)



axes[0].text(-0.15,1.15,'A',fontsize=18,transform=axes[0].transAxes,fontweight='bold',va='top',ha='right')
axes[1].text(-0.15,1.15,'B',fontsize=18,transform=axes[1].transAxes,fontweight='bold',va='top',ha='right')



plt.tight_layout()

cur_ax = axes[1]
x_pos = cur_ax.get_position().intervalx
y_pos = cur_ax.get_position().intervaly
cur_ax.set_position([x_pos[0]+0.02, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])


if save_flag:
    plt.savefig('./Figures/SuppFig4_TimeVarying_ns.jpg',dpi=dpi_val,bbox_inches='tight')  
