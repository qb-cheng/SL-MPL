# -*- coding: utf-8 -*-
"""
Compare the RMSE performance between the MPL-based estimator and the extended
estimator that additionally considers limited sampling effect in its model.
Also plot the empirical distribution with fixed trajectory length T as examples.
"""

import pickle,Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import pandas as pd
import seaborn as sns

save_flag = True
dpi_val = 350

FontSize = 16
MarkerSize = 4
lw = 1.5

dt = 10

ymin = 5e-4
ymax = 0.5

Sets = np.array([11,41])
Ts = [20,50,100,150,300,450]
ns_ratios = [0.01,0.1]
T_interest = 150


fig,axes = plt.subplots(len(Sets),len(ns_ratios)*2,figsize=(15,6),dpi=dpi_val)
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
    t = np.arange(0,T,dt)
        
    for ns_idx,ns_ratio in enumerate(ns_ratios):
        ns = int(N * ns_ratio)
        
        df_est = pd.DataFrame()
        estimates = []
        EstName = []
        
            
        with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
            StoTrajs = pickle.load(f)
        
        
        # Compute the empirical estimator variance under joint effects of finite sampling and genetic drift
        sFSRMSE = np.zeros(len(t)-2)
        SampledStoTrajs = Functions.SampleOnce(StoTrajs[t], ns)
        sFS_joint = Functions.SLMPL(SampledStoTrajs, dt, ns, u)
        for idx in range(sFS_joint.shape[0]):
            CurEsts = sFS_joint[idx,:]
            CurEsts = CurEsts[np.isfinite(CurEsts)]
            sFSRMSE[idx] = np.sqrt(np.mean((CurEsts - s)**2))
            
        # Compute the empirical estimator variance of s_HMM
        EstHMMRMSE = np.zeros(len(Ts))
        for T_idx,CurT in enumerate(Ts):
            HMMPkl = './ModelComp/ns_ratio_'+str(ns_ratio)+'/N'+str(N)+'_T'+str(CurT)+'_HMM.pkl'
            with open(HMMPkl,'rb') as f:
                s_MPL,TimeCost_MPL,s_HMM,TimeCost_sHMM,info = pickle.load(f)
            s_HMM = s_HMM[np.isfinite(s_HMM)]
            EstHMMRMSE[T_idx] = np.sqrt(np.mean((s_HMM - s)**2))
            
            if CurT == T_interest:
                estimates += list(s_MPL)
                EstName += ['$\\^s$']*len(s_MPL)
            
                estimates += list(s_HMM)
                EstName += ['$\\^s_{\\mathrm{LS}}$']*len(s_HMM)
                
        # # Compute the empirical estimator variance of s_QS
        # EstQSRMSE = np.zeros(len(Ts))
        # for T_idx,CurT in enumerate(Ts):
        #     DetPkl = './ModelComp_1kTrajs/ns_ratio_'+str(ns_ratio)+'/N'+str(N)+'_T'+str(CurT)+'_det.pkl'
        #     with open(DetPkl,'rb') as f:
        #         s_FS,TimeCost_sFS,s_QS,TimeCost_sQS,info = pickle.load(f)
        #     s_QS = s_QS[np.isfinite(s_QS)]
        #     EstQSRMSE[T_idx] = np.sqrt(np.mean((s_QS - s)**2))
            
        #     if CurT == T_interest:            
        #         estimates += list(s_QS)
        #         EstName += ['$\\^s_{\\mathrm{QS}}$']*len(s_QS)
        
        
        cur_ax = axes[set_idx,ns_idx]
        if set_idx == 0:
            cur_ax.set_title('$n_s$ / $N = $' + str(ns_ratio), fontsize=FontSize, pad=10)
        
        if ns_idx == 0:
            cur_ax.set_ylabel('$N = $' + str(N), fontsize=FontSize)
        # Plot the estimator RMSE
        cur_ax.semilogy(t[2:],sFSRMSE,label='$\\^s$',color='C1',linewidth=lw,zorder=1)
        cur_ax.semilogy(Ts,EstHMMRMSE,label='$\\^s_{\\mathrm{LS}}$',color='C4',linestyle='--',marker='o',ms=MarkerSize,zorder=2)
        if ns_idx > 0:
            cur_ax.axes.get_yaxis().set_ticklabels([])
            
        if set_idx == len(Sets)-1:
            cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
        else:
            cur_ax.axes.get_xaxis().set_ticklabels([])
            
        cur_ax.set_xlim((0,T+25))
        cur_ax.set_ylim((ymin, ymax))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
        
        
        cur_ax = axes[set_idx,ns_idx+len(Sets)]
        df_est['Estimate'] = estimates
        df_est['EstimatorName'] = EstName
        violin = sns.violinplot(data=df_est,x="EstimatorName",y="Estimate",inner="quart", linewidth=lw, ax=cur_ax, palette={'$\\^s$':'C1','$\\^s_{\\mathrm{LS}}$':'C4'},legend=False)
        for patch in violin.collections:
            patch.set_alpha(0.7)
        if set_idx == len(Sets)-1:
            cur_ax.set_xlabel("Estimator",fontsize=FontSize)
        else:
            cur_ax.set_xlabel("",fontsize=FontSize)
            cur_ax.axes.get_xaxis().set_ticklabels([])
        
        if ns_idx == 0:
            # cur_ax.text(-0.65, 0.5, '$N = $' + str(N), ha='center', va='center', fontsize=FontSize, transform=cur_ax.transAxes)
            # cur_ax.set_ylabel("Selection coefficient \n estimate",fontsize=FontSize)
            cur_ax.set_ylabel('$N = $' + str(N),fontsize=FontSize)
        else:
            cur_ax.set_ylabel("",fontsize=FontSize)
            cur_ax.axes.get_yaxis().set_ticklabels([])
            
        cur_ax.axhline(y = s, color ='gray', linestyle ="--",linewidth=lw,label='$s$')
        cur_ax.set_ylim((-0.06,0.1))
        cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.04))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        if set_idx == 0:
            cur_ax.set_title('$n_s$ / $N = $' + str(ns_ratio), fontsize=FontSize, pad=10)
        
axes[0,0].legend(fontsize=FontSize,frameon=False,loc=1)
plt.tight_layout()

for set_idx,thisSet in enumerate(Sets):
    cur_ax = axes[set_idx,0]
    x_pos = cur_ax.get_position().intervalx
    y_pos = cur_ax.get_position().intervaly
    cur_ax.set_position([x_pos[0]+0.02, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
    
    cur_ax = axes[set_idx,1]
    x_pos = cur_ax.get_position().intervalx
    y_pos = cur_ax.get_position().intervaly
    cur_ax.set_position([x_pos[0]-0.05, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
    
    cur_ax = axes[set_idx,2]
    x_pos = cur_ax.get_position().intervalx
    y_pos = cur_ax.get_position().intervaly
    cur_ax.set_position([x_pos[0]+0.07, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])

fig.text(0.02, 0.5,"Root mean square error (RMSE)", ha='center', va='center', fontsize=FontSize, rotation='vertical')      
fig.text(0.56, 0.5,"Selection coefficient estimate", ha='center', va='center', fontsize=FontSize, rotation='vertical')    
fig.text(0.018,1.02,'A',fontsize=18,fontweight='bold',va='top',ha='center')
fig.text(0.558,1.02,'B',fontsize=18,fontweight='bold',va='top',ha='center')

if save_flag:
    plt.savefig('./Figures/Fig6_RMSE.jpg',dpi=dpi_val,bbox_inches='tight')