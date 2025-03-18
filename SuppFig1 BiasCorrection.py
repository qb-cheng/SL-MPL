# -*- coding: utf-8 -*-
"""
Show that the bias-correction factor on the denominator of the MPL-based 
estimator \hat{s} helps center the empirical distribution at the true selection coefficient s.
"""

import Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
import pandas as pd


save_flag = True
dpi_val = 350
FontSize = 14
MarkerSize = 2
lw = 1.5

ymin = -0.02
ymax = 0.06

thisSet = 11
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)

dt = 10
t = np.arange(0,T,dt)

lambda_nss = [10,20]
T_interests = [150,300,450]
MCRun = 10000



SampleSizes = np.zeros((len(t),MCRun,len(lambda_nss)))

Palette = sns.color_palette('Set2')


DeterTraj = Functions.DeterministicTrajectory(s, x0, T, u)
ObTraj_examples = np.zeros((len(t),len(lambda_nss)))

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
            cur_t = t[t <= cur_T]
            cur_dt = cur_t[1:] - cur_t[:-1]
            
            seg_cur_ObTraj = cur_ObTraj[t <= cur_T]
            seg_cur_ns = cur_ns[t <= cur_T]
            
            Dhat = seg_cur_ObTraj[-1] - seg_cur_ObTraj[0] - u*np.sum(cur_dt*(1-2*seg_cur_ObTraj[:-1]))
            Vhat = np.sum(cur_dt * seg_cur_ObTraj[:-1] * (1 - seg_cur_ObTraj[:-1]))
            Vhat_corrected = np.sum(cur_dt * seg_cur_ObTraj[:-1] * (1 - seg_cur_ObTraj[:-1]) * seg_cur_ns[:-1] / (seg_cur_ns[:-1]-1))
            
            s_MPL = Dhat / Vhat
            s_FS = Dhat / Vhat_corrected
            
            estimates.append(s_FS)
            est_type.append('$\^s$')
            TrajLen.append(cur_T)
            SampleSize_Lambda.append('$\overline{n}_s = $'+str(lambda_ns))
            
            estimates.append(s_MPL)
            est_type.append('$\^s_{\mathrm{MPL}}$')
            TrajLen.append(cur_T)
            SampleSize_Lambda.append('$\overline{n}_s = $'+str(lambda_ns))
            
    
    ObTraj_examples[:,ns_idx] = cur_ObTraj
    

df_est['estimates'] = estimates
df_est['est_type'] = est_type
df_est['Trajectory length'] = TrajLen
df_est['Sample size'] = SampleSize_Lambda

df_est = df_est[np.isfinite(df_est["estimates"])]
   


fig,axes = plt.subplots(1,len(lambda_nss),figsize=(10,4),dpi=dpi_val)

for ns_idx,cur_ns in enumerate(lambda_nss):
    cur_ax = axes[ns_idx]
    sns.violinplot(data=df_est[df_est['Sample size'] == '$\overline{n}_s = $'+str(cur_ns)], x='Trajectory length', y='estimates', hue='est_type', split=True, inner="quart", linewidth=lw, ax=cur_ax, palette=[Palette[0],Palette[4]], cut=0, alpha=0.8, scale='area')

    cur_ax.axhline(y=s, color ='gray', linestyle ="dotted",linewidth=lw,label='True selection\ncoefficient $s$')
    cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
    
    cur_ax.set_ylim((ymin,ymax))
    cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    cur_ax.tick_params(axis='x', labelsize=FontSize)
    cur_ax.tick_params(axis='y', labelsize=FontSize)
    cur_ax.text(0.9,0.9,'$\overline{n}_s = $'+str(cur_ns),fontsize=FontSize,ha='right',va='center',transform=cur_ax.transAxes)
    if ns_idx == 0:
        cur_ax.set_ylabel('Selection coefficient estimate',fontsize=FontSize)
    else:
        cur_ax.set_ylabel('',fontsize=FontSize)
        cur_ax.axes.get_yaxis().set_ticklabels([])
    if ns_idx == len(lambda_nss)-1:
        cur_ax.legend(fontsize=FontSize,frameon=False,loc=4)
        handles, labels = cur_ax.get_legend_handles_labels()
        legend1 = plt.legend(handles[:2], labels[:2], loc='lower center', bbox_to_anchor=(0.35,0), fontsize=FontSize, frameon=False)
        legend2 = plt.legend(handles[2:], labels[2:], loc='lower right', bbox_to_anchor=(1,0), fontsize=FontSize, frameon=False)
        cur_ax.add_artist(legend1)
    else:
        cur_ax.legend_.remove()
    
    



plt.tight_layout()


if save_flag:
    plt.savefig('./Figures/SuppFig1_BiasCorrection.jpg',dpi=dpi_val,bbox_inches='tight')  