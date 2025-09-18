# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 16:26:13 2025

@author: cerul
"""

import pickle,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
# import matplotlib.image as mpimg
from PyPDF2 import PdfMerger


save_flag = True
dpi_val = 600

FontSize = 16
MarkerSize = 4
lw = 1.5

dt = 10

ymin = 5e-4
ymax = 0.5

Sets = np.array([11,41])
ns_ratios = [0.01,0.1]
T_interests = [150,100,60]
BarWidth = 0.3
dt_max = 50

for T_idx,curT in enumerate(T_interests):
    dts = np.array([factor for factor in range (1,curT+1) if curT % factor == 0 and factor <= dt_max and factor >= 10])
    
    fig,axes = plt.subplots(len(Sets),len(ns_ratios)*2,figsize=(15,5.5),dpi=dpi_val)
    for set_idx,thisSet in enumerate(Sets):
        N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
        
        for ns_idx,ns_ratio in enumerate(ns_ratios):
            ns = int(N * ns_ratio)
            
            # Compute the empirical estimator variance under joint effects of finite sampling and genetic drift
            sFSRMSE = np.zeros(len(dts))
            EstHMMRMSE = np.zeros(len(dts))
            ExeTime = np.zeros((len(dts),2))
            for dt_idx,cur_dt in enumerate(dts):
                HMMPkl = './ModelComp/ns_ratio_'+str(ns_ratio)+'/N'+str(N)+'_T'+str(curT)+'_dt'+str(cur_dt)+'_HMM.pkl'
                with open(HMMPkl,'rb') as f:
                    s_MPL,TimeCost_MPL,s_HMM,TimeCost_sHMM,info = pickle.load(f)
                s_MPL = s_MPL[np.isfinite(s_MPL)]
                sFSRMSE[dt_idx] = np.sqrt(np.mean((s_MPL - s)**2))
                ExeTime[dt_idx,0] = TimeCost_MPL
                s_HMM = s_HMM[np.isfinite(s_HMM)]
                EstHMMRMSE[dt_idx] = np.sqrt(np.mean((s_HMM - s)**2))
                ExeTime[dt_idx,1] = TimeCost_sHMM
            
            cur_ax = axes[set_idx,ns_idx]
            if set_idx == 0:
                cur_ax.set_title('$n_s$ / $N = $' + str(ns_ratio), fontsize=FontSize, pad=10)
            
            if ns_idx == 0:
                cur_ax.set_ylabel('$N = $' + str(N), fontsize=FontSize)
            # Plot the estimator RMSE
            cur_ax.semilogy(dts,sFSRMSE,label='$\\^s$',color='C1',linewidth=lw,marker='o',ms=MarkerSize,zorder=1)
            cur_ax.semilogy(dts,EstHMMRMSE,label='$\\^s_{\\mathrm{LS}}$',color='C4',linestyle='--',marker='o',ms=MarkerSize,zorder=2)
            if ns_idx > 0:
                cur_ax.axes.get_yaxis().set_ticklabels([])
                
            if set_idx == len(Sets)-1:
                cur_ax.set_xlabel('Time sampling step, $\Delta t$',fontsize=FontSize)
            else:
                cur_ax.axes.get_xaxis().set_ticklabels([])
                
            cur_ax.set_xlim((8,dt_max + 2))
            cur_ax.set_ylim((ymin, ymax))
            cur_ax.tick_params(axis='x', labelsize=FontSize)
            cur_ax.tick_params(axis='y', labelsize=FontSize)
            cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            
            
            cur_ax = axes[set_idx,ns_idx+len(Sets)]
            cur_ax.bar(np.arange(len(dts))-BarWidth/2-0.01,ExeTime[:, 0], width=BarWidth, label='$\\^s$', color='C1')
            cur_ax.bar(np.arange(len(dts))+BarWidth/2+0.01,ExeTime[:, 1], width=BarWidth, label='$\\^s_{\\mathrm{LS}}$', color='C4')
            cur_ax.set_yscale('log')
            if set_idx == len(Sets)-1:
                cur_ax.set_xlabel("Time sampling step, $\Delta t$",fontsize=FontSize)
                cur_ax.set_xticks(np.arange(len(dts)))
                cur_ax.set_xticklabels([str(cur_dt) for cur_dt in dts])
    
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
            
            
            cur_ax.set_ylim((1e-6,50))
            cur_ax.set_xlim((-0.5,len(dts)-0.5))
            cur_ax.tick_params(axis='x', labelsize=FontSize)
            cur_ax.tick_params(axis='y', labelsize=FontSize)
            if set_idx == 0:
                cur_ax.set_title('$n_s$ / $N = $' + str(ns_ratio), fontsize=FontSize, pad=10)
        
    axes[0,0].legend(fontsize=FontSize,frameon=False,loc=1)
    axes[0,2].legend(fontsize=FontSize,frameon=False,loc=1,ncol=2,bbox_to_anchor=(0.95, 1.05))
    plt.tight_layout()
    
    for set_idx,thisSet in enumerate(Sets):
        cur_ax = axes[set_idx,0]
        x_pos = cur_ax.get_position().intervalx
        y_pos = cur_ax.get_position().intervaly
        cur_ax.set_position([x_pos[0]+0.02, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
        
        cur_ax = axes[set_idx,1]
        x_pos = cur_ax.get_position().intervalx
        y_pos = cur_ax.get_position().intervaly
        cur_ax.set_position([x_pos[0]-0.03, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
        
        cur_ax = axes[set_idx,2]
        x_pos = cur_ax.get_position().intervalx
        y_pos = cur_ax.get_position().intervaly
        cur_ax.set_position([x_pos[0]+0.04, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
        
        cur_ax = axes[set_idx,3]
        x_pos = cur_ax.get_position().intervalx
        y_pos = cur_ax.get_position().intervaly
        cur_ax.set_position([x_pos[0]-0.01, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])
    
    fig.text(-0.07, 0.5,"$T = $"+str(curT), ha='left', va='center', fontsize=FontSize+2)    
    fig.text(0.02, 0.5,"Root mean square error (RMSE)", ha='center', va='center', fontsize=FontSize, rotation='vertical')      
    fig.text(0.537, 0.5,"Mean execution time (in seconds)", ha='center', va='center', fontsize=FontSize, rotation='vertical')    
    fig.text(0.018,1,chr(ord('A')+T_idx*2),fontsize=18,fontweight='bold',va='top',ha='center')
    fig.text(0.535,1,chr(ord('B')+T_idx*2),fontsize=18,fontweight='bold',va='top',ha='center')
    
    if save_flag:
        plt.savefig('./Figures/HMM_TradeOffs_T'+str(curT)+'.pdf',dpi=dpi_val,bbox_inches='tight')

if save_flag:        
    # Load saved figures
    merger = PdfMerger()
    for T_idx,curT in enumerate(T_interests):
        merger.append('./Figures/HMM_TradeOffs_T'+str(curT)+'.pdf')

    # Write out the combined PDF
    merger.write('./Figures/SuppFig11_HMM_TradeOffs.pdf')
    merger.close()

    
# if save_flag:        
    # Load saved figures
    # images = []
    # for T_idx,curT in enumerate(T_interests):
        # CurImg = mpimg.imread('./Figures/HMM_TradeOffs_T'+str(curT)+'.jpg')
        # if T_idx > 0:
        #     CurImgSize = np.array(CurImg.shape)
        #     CurImgSize[0] = np.round(CurImgSize[0]*0.1)
        #     SpaceImg = np.ones(CurImgSize).astype('uint8')*255
        #     images.append(SpaceImg)
        # images.append(CurImg)

    # Stack images vertically
    # stacked_image = np.vstack(images)
    
    # Save the stacked image
    # plt.imsave('./Figures/SuppFig11_HMM_TradeOffs.pdf', stacked_image, dpi=dpi_val)        

