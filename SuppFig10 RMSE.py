# -*- coding: utf-8 -*-
"""
The RMSE performance comparison between the MPL-based estimator and the extended
estimator that additionally considers limited sampling effect in its model.
"""


import pickle,Functions,GetCaseArg,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 

save_flag = True
dpi_val = 350

FontSize = 14
MarkerSize = 4
lw = 1.5

dt = 10

ymin = 1e-4
ymax = 0.9

Sets = np.array([11,21,31,41,51,61,71])
Ts = [20,50,100,150,300,450]
ns_ratios = [0.01,0.02,0.05,0.1,0.2]

FirstTimeLegend = True

fig,axes = plt.subplots(len(Sets),len(ns_ratios),figsize=(15,17),dpi=dpi_val)
for set_idx,thisSet in enumerate(Sets):
    N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
    t = np.arange(0,T,dt)
        
    for ns_idx,ns_ratio in enumerate(ns_ratios):
        ns = int(N * ns_ratio)
        
        cur_ax = axes[set_idx,ns_idx]
        if set_idx == 0:
            cur_ax.set_title('$n_s$ / $N = $' + str(ns_ratio), fontsize=FontSize, pad=10)
        
        if ns_idx == 0:
            cur_ax.text(-0.7, 0.5, '$N = $' + str(N), ha='center', va='center', fontsize=FontSize, transform=cur_ax.transAxes)
        
        DetPkl = './ModelComp/ns_ratio_'+str(ns_ratio)+'/N'+str(N)+'_T150_det.pkl'
        HMMPkl = './ModelComp/ns_ratio_'+str(ns_ratio)+'/N'+str(N)+'_T150_HMM.pkl'
        if (not os.path.exists(DetPkl)) or (not os.path.exists(HMMPkl)):
            cur_ax.axis("off")
            continue
            
        with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
            StoTrajs = pickle.load(f)
        
        
        
        # Compute the empirical estimator variance under joint effects of finite sampling and genetic drift
        sMPLRMSE = np.zeros(len(t)-2)
        SampledStoTrajs = Functions.SampleOnce(StoTrajs[t], ns)
        sMPL_joint = Functions.SLMPL(SampledStoTrajs, dt, ns, u)
        for idx in range(sMPL_joint.shape[0]):
            CurEsts = sMPL_joint[idx,:]
            CurEsts = CurEsts[np.isfinite(CurEsts)]
            sMPLRMSE[idx] = np.sqrt(np.mean((CurEsts - s)**2))
            
        # Compute the empirical estimator variance of s_HMM
        EstHMMRMSE = np.zeros(len(Ts))
        for T_idx,CurT in enumerate(Ts):
            HMMPkl = './ModelComp/ns_ratio_'+str(ns_ratio)+'/N'+str(N)+'_T'+str(CurT)+'_HMM.pkl'
            with open(HMMPkl,'rb') as f:
                _,_,s_HMM,TimeCost_sHMM,info = pickle.load(f)
            s_HMM = s_HMM[np.isfinite(s_HMM)]
            EstHMMRMSE[T_idx] = np.sqrt(np.mean((s_HMM - s)**2))
            
        """
        # Compute the empirical estimator variance of s_Det
        EstDetRMSE = np.zeros(len(Ts))
        for T_idx,CurT in enumerate(Ts):
            DetPkl = './ModelComp/ns_ratio_'+str(ns_ratio)+'/N'+str(N)+'_T'+str(CurT)+'_det.pkl'
            with open(DetPkl,'rb') as f:
                _,_,s_Det,TimeCost_sDet,info = pickle.load(f)
            s_Det = s_Det[np.isfinite(s_Det)]
            EstDetRMSE[T_idx] = np.sqrt(np.mean((s_Det - s)**2))
        """
        
        # Plot the estimator variances
        line1 = cur_ax.semilogy(t[2:],sMPLRMSE,label='$\\^s$',color='C1',linewidth=lw,zorder=1)
        line2 = cur_ax.semilogy(Ts,EstHMMRMSE,label='$\\^s_{\\mathrm{LS}}$',color='C4',linestyle='--',marker='o',ms=MarkerSize,zorder=2)
        # line3 = cur_ax.semilogy(Ts,EstDetRMSE,label='$\\^s_{\\mathrm{QS}}$',color='C2',linestyle='--',marker='x',ms=MarkerSize+2,zorder=3)
        # cur_ax.axhline(y = s, color ="gray", linestyle ="dotted",linewidth=lw) # ,label='True selection\ncoefficient'
        if ns_idx == 0:
            cur_ax.set_ylabel('Root mean square\nerror (RMSE)',fontsize=FontSize)
        else:
            cur_ax.axes.get_yaxis().set_ticklabels([])
            
        if set_idx == len(Sets)-1:
            cur_ax.set_xlabel('Trajectory length, $T$',fontsize=FontSize)
        else:
            cur_ax.axes.get_xaxis().set_ticklabels([])
            
        if FirstTimeLegend:
            # legend1 = cur_ax.legend(handles=[line1[0]],labels=['$\\^s$'],fontsize=FontSize,frameon=False,loc=4,bbox_to_anchor=(0.5,0.075))
            # cur_ax.add_artist(legend1)
            # cur_ax.legend(handles=[line2[0],line3[0]],labels=['$\\^s_{\\mathrm{LS}}$','$\\^s_{\\mathrm{QS}}$'],fontsize=FontSize,frameon=False,loc=4)
            cur_ax.legend(fontsize=FontSize,frameon=False,loc=4)
            FirstTimeLegend = False
            
        cur_ax.set_xlim((0,T+25))
        cur_ax.set_ylim((ymin, ymax))
        cur_ax.tick_params(axis='x', labelsize=FontSize)
        cur_ax.tick_params(axis='y', labelsize=FontSize)
        cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
        

        
        
plt.tight_layout()
plt.subplots_adjust(wspace=1.5/15,hspace=1.5/17)

if save_flag:
    plt.savefig('./Figures/SuppFig10_EstComp_RMSE.jpg',dpi=dpi_val,bbox_inches='tight')  