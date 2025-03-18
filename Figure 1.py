# -*- coding: utf-8 -*-
"""
Plot the performance of the MPL-based estimator under limited sampling effect.
"""

import Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
import pandas as pd


save_flag = True
dpi_val = 350
FontSize = 16
MarkerSize = 2
lw = 1.5

xmin = -0.02
xmax = 0.06

thisSet = 11
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)

dt = 10
t = np.arange(0,T,dt)

lambda_nss = [10,20]
T_interests = [151,301,451]
MCRun = 10000



SampleSizes = np.zeros((len(t),MCRun,len(lambda_nss)))

Palette = sns.color_palette('Set2')

fig,axes = plt.subplots(1,2,figsize=(15,4),dpi=dpi_val)

DeterTraj = Functions.DeterministicTrajectory(s, x0, T, u)
ObTraj_examples = np.zeros((len(t),len(lambda_nss)))

axes[0].plot(t,DeterTraj[t],linewidth=lw,color='gray',label='Population')

df_est = pd.DataFrame()
estimates = []
est_type = []
TrajLen = []
SampleSize_Lambda = []

for ns_idx,lambda_ns in enumerate(lambda_nss):
    for MCidx in range(MCRun):
        cur_ns = np.random.poisson(lambda_ns, len(t))
        while any(cur_ns < 2):
            cur_ns = np.random.poisson(lambda_ns, len(t))
        SampleSizes[:,MCidx,ns_idx] = cur_ns

        cur_ObTraj = np.zeros(len(t))
        for t_idx in range(len(t)):
            cur_ObTraj[t_idx] = np.random.binomial(cur_ns[t_idx], DeterTraj[t[t_idx]]) / cur_ns[t_idx]
            
        for T_idx,cur_T in enumerate(T_interests):
            cur_t = t[t < cur_T]
            cur_dt = cur_t[1:] - cur_t[:-1]
            
            seg_cur_ObTraj = cur_ObTraj[t < cur_T]
            seg_cur_ns = cur_ns[t < cur_T]
            
            Dhat = seg_cur_ObTraj[-1] - seg_cur_ObTraj[0] - u*np.sum(cur_dt*(1-2*seg_cur_ObTraj[:-1]))
            Vhat = np.sum(cur_dt * seg_cur_ObTraj[:-1] * (1 - seg_cur_ObTraj[:-1]))
            Vhat_corrected = np.sum(cur_dt * seg_cur_ObTraj[:-1] * (1 - seg_cur_ObTraj[:-1]) * seg_cur_ns[:-1] / (seg_cur_ns[:-1]-1))
            
            s_MPL = Dhat / Vhat
            s_FS = Dhat / Vhat_corrected
            
            estimates.append(s_FS)
            est_type.append('$\^s$')
            TrajLen.append(cur_T-1)
            SampleSize_Lambda.append('$\overline{n}_s = $'+str(lambda_ns))
            
            estimates.append(s_MPL)
            est_type.append('$\^s_{\mathrm{MPL}}$')
            TrajLen.append(cur_T-1)
            SampleSize_Lambda.append('$\overline{n}_s = $'+str(lambda_ns))
            
    
    ObTraj_examples[:,ns_idx] = cur_ObTraj
    axes[0].plot(cur_t,cur_ObTraj,label='Observations ($\overline{n}_s = $'+str(lambda_ns)+')',color=Palette[ns_idx],linewidth=lw,zorder=1)
    

df_est['estimates'] = estimates
df_est['est_type'] = est_type
df_est['Trajectory length'] = TrajLen
df_est['Sample size'] = SampleSize_Lambda

df_est = df_est[np.isfinite(df_est["estimates"])]
   
axes[0].set_xlim((0,T+9))
axes[0].set_ylim((-0.05,1.05))
axes[0].tick_params(axis='x', labelsize=FontSize)
axes[0].tick_params(axis='y', labelsize=FontSize)   
axes[0].set_xlabel('Generation, $t$',fontsize=FontSize)
axes[0].set_ylabel('Mutant allele frequency',fontsize=FontSize)
axes[0].legend(fontsize=FontSize,loc=4,frameon=False)
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(150))
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))    
    
    
    
    
violin = sns.violinplot(data=df_est[df_est['est_type'] == '$\^s$'], x='Trajectory length', y='estimates', hue='Sample size', split=True, inner="quart", linewidth=lw, ax=axes[1], palette=Palette[:2], cut=0, alpha=0.8, scale='area')

axes[1].axhline(y=s, color ='gray', linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$')
axes[1].set_xlabel('Trajectory length, $T$',fontsize=FontSize)
axes[1].set_ylabel('Selection coefficient\nestimate, $\\^s$',fontsize=FontSize)
axes[1].set_ylim((xmin,xmax))
axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.02))
axes[1].tick_params(axis='x', labelsize=FontSize)
axes[1].tick_params(axis='y', labelsize=FontSize)
axes[1].legend(fontsize=FontSize,frameon=False,loc=4)#,ncol=2

handles, labels = axes[1].get_legend_handles_labels()
legend1 = plt.legend(handles[:2], labels[:2], loc='lower center', bbox_to_anchor=(0.35,0), fontsize=FontSize, frameon=False)
legend2 = plt.legend(handles[2:], labels[2:], loc='lower right', bbox_to_anchor=(1,0), fontsize=FontSize, frameon=False)
axes[1].add_artist(legend1)


plt.tight_layout()
plt.subplots_adjust(wspace=0.4)


if save_flag:
    plt.savefig('./Figures/Fig1_EstPerf.jpg',dpi=dpi_val,bbox_inches='tight')  