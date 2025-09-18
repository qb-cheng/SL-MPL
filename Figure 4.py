# -*- coding: utf-8 -*-
"""
Plot the KL divergence with different values of sample size ns and trajectory length T,
in the deterministic scenario under the effect of limited sampling.
Show the speed of the MPL-based estimator converging to a Gaussian distribution.
"""

import Functions,GetCaseArg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import seaborn as sns


save_flag = True

dpi_val = 600

AlphaVal = 0.4
FontSize = 16
lw = 1.5



ns_ref = 10
T_ref = 151
ns_larger = 20
T_larger = 301

ymax = 200
ystep = 50
BinsNum = 50
xmin = -0.01
xmax = 0.05
xstep = 0.02
bins = np.linspace(xmin,xmax,BinsNum)

Palette = sns.color_palette('Set2')

u = 1e-3
dt = 10
MCRun = 1000000
KL_resol = 0.01


thisSet = 11
N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
DeterTraj = Functions.DeterministicTrajectory(s,x0,T,u)

t = np.arange(0,T,dt)
DeterIV = Functions.ComputeIV(DeterTraj[t], dt)
Vs = np.array([int(np.rint(DeterIV[int(T_ref/dt)-2])),int(np.rint(DeterIV[int(T_larger/dt)-2]))])



fig,axes = plt.subplots(1,3,figsize=(15,4.5),dpi=dpi_val)            

# KL divergence plot for the deterministic trajectory
nss = [10,20,50]
t = np.arange(0,T,dt)
Ts = np.arange(dt*2,T,dt)
KL_div = np.zeros((len(DeterIV),len(nss))) # KL divergence for all simulated trajectories


for ns_idx,cur_ns in enumerate(nss):
    SampledTrajs = Functions.SampleTrajectory(DeterTraj[t], cur_ns, MCRun)
    
    ests_All = Functions.SLMPL(SampledTrajs, dt, cur_ns, u)
    for T_idx,CurEsts in enumerate(ests_All):
        CurEsts = CurEsts[np.isfinite(CurEsts)]
        KL_div[T_idx,ns_idx] = Functions.KL_div(CurEsts, KL_resol)



cur_ax = axes[0]
# print(cur_ax.get_position())
# cur_ax.set_position([0.125, 0.125, 0.225, 0.88])
for ns_idx,cur_ns in enumerate(nss):
    # cur_ax.plot(t[2:],KL_div[:,ns_idx],linewidth=lw,color=CustomizedColor[ns_idx],label='$n_s$ = '+str(cur_ns))
    cur_ax.plot(DeterIV,KL_div[:,ns_idx],linewidth=lw,color=Palette[ns_idx],label='$n_s$ = '+str(cur_ns))

cur_ax.legend(fontsize=FontSize,frameon=False,loc=1) #,bbox_to_anchor=(1, 0.85)
# cur_ax.set_xlabel('Generations used for inference, $T$',fontsize=FontSize)
# cur_ax.set_xlabel('Generations used for inference, $T$'+'\n'+'In parenthesis: Integrated variance, $V$',fontsize=FontSize)
cur_ax.set_xlabel('Integrated variance, $V$',fontsize=FontSize)
cur_ax.set_ylabel('KL divergence',fontsize=FontSize)
cur_ax.set_xlim((0,np.max(DeterIV)+2))
cur_ax.set_ylim((0,3))
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(1))






# Plot examples of empirical distributions

t = np.arange(0,T_ref,dt)
ObservableTraj = DeterTraj[t]


SampledTrajs = Functions.SampleTrajectory(ObservableTraj, ns_ref, MCRun)
D_hat = SampledTrajs[-1,:] - SampledTrajs[0,:] - u*dt*np.sum(1-2*SampledTrajs[:-1,:],axis=0)
V_hat = dt * np.sum(SampledTrajs[:-1,:]*(1-SampledTrajs[:-1,:]),axis=0) / (1-1/ns_ref)
s_FS_ref = D_hat / V_hat
s_FS_ref = s_FS_ref[np.isfinite(s_FS_ref)]

ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(ObservableTraj,dt,ns_ref,u)
AnalyticalVar_ref = ((VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2))[-1]


SampledTrajs = Functions.SampleTrajectory(ObservableTraj, ns_larger, MCRun)
D_hat = SampledTrajs[-1,:] - SampledTrajs[0,:] - u*dt*np.sum(1-2*SampledTrajs[:-1,:],axis=0)
V_hat = dt * np.sum(SampledTrajs[:-1,:]*(1-SampledTrajs[:-1,:]),axis=0) / (1-1/ns_larger)
s_FS_largerns = D_hat / V_hat
s_FS_largerns = s_FS_largerns[np.isfinite(s_FS_largerns)]

ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(ObservableTraj,dt,ns_larger,u)
AnalyticalVar_ns_larger = ((VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2))[-1]



cur_ax = axes[1]
cur_ax.hist(s_FS_ref,bins=bins,alpha=AlphaVal,density=True,color=Palette[0],edgecolor='black',linewidth=lw/2,label='$n_s = $'+str(ns_ref))
# cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),np.mean(s_FS_ref),np.std(s_FS_ref)),linewidth=2,color=Palette[0])
cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),s,np.sqrt(AnalyticalVar_ref)),linewidth=2,color=Palette[0])
cur_ax.hist(s_FS_largerns,bins=bins,alpha=AlphaVal,density=True,color=Palette[1],edgecolor='black',linewidth=lw/2,label='$n_s = $'+str(ns_larger),zorder=3)
# cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),np.mean(s_FS_largerns),np.std(s_FS_largerns)),linewidth=2,color=Palette[1])
cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),s,np.sqrt(AnalyticalVar_ns_larger)),linewidth=2,color=Palette[1])
cur_ax.plot(np.arange(-10,-9),0,linewidth=2,color='gray',label='Analytical\nGaussian')
cur_ax.set_xlim((xmin,xmax))
cur_ax.set_ylim((0,ymax))
cur_ax.set_xlabel('Selection coefficient estimate, $\\^s$',fontsize=FontSize)
cur_ax.set_ylabel('Density',fontsize=FontSize)
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))
cur_ax.legend(fontsize=FontSize,frameon=False,loc=2)

cur_ax.text(0.95,0.85,'$T = $' + str(T_ref-1) + '\n ($V = $'+str(Vs[0]) + ')',ha='right',va='center',fontsize=FontSize, transform=cur_ax.transAxes)



t = np.arange(0,T_larger,dt)
ObservableTraj = DeterTraj[t]
SampledTrajs = Functions.SampleTrajectory(ObservableTraj, ns_ref, MCRun)
D_hat = SampledTrajs[-1,:] - SampledTrajs[0,:] - u*dt*np.sum(1-2*SampledTrajs[:-1,:],axis=0)
V_hat = dt * np.sum(SampledTrajs[:-1,:]*(1-SampledTrajs[:-1,:]),axis=0) / (1-1/ns_ref)
s_FS_largerT = D_hat / V_hat
s_FS_largerT = s_FS_largerT[np.isfinite(s_FS_largerT)]

ED,EV,VarD,CovDV,VarV = Functions.TaylorTerms(ObservableTraj,dt,ns_ref,u)
AnalyticalVar_T_larger = ((VarD/EV**2 - 2*(ED/EV)*CovDV/EV**2 + (ED/EV)**2*VarV/EV**2))[-1]


cur_ax = axes[2]
cur_ax.hist(s_FS_ref,bins=bins,alpha=AlphaVal,density=True,color=Palette[0],edgecolor='black',linewidth=lw/2,label='$T = $'+str(T_ref-1)+' ($V = $'+str(Vs[0])+')')
# cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),np.mean(s_FS_ref),np.std(s_FS_ref)),linewidth=2,color=Palette[0])
cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),s,np.sqrt(AnalyticalVar_ref)),linewidth=2,color=Palette[0])
cur_ax.hist(s_FS_largerT,bins=bins,alpha=AlphaVal,density=True,color=Palette[6],edgecolor='black',linewidth=lw/2,label='$T = $'+str(T_larger-1)+' ($V = $'+str(Vs[1])+')',zorder=3)
# cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),np.mean(s_FS_largerT),np.std(s_FS_largerT)),linewidth=2,color=Palette[5])
cur_ax.plot(np.linspace(xmin,xmax,BinsNum*10),stats.norm.pdf(np.linspace(xmin,xmax,BinsNum*10),s,np.sqrt(AnalyticalVar_T_larger)),linewidth=2,color=Palette[6])
cur_ax.plot(np.arange(-10,-9),0,linewidth=2,color='gray',label='Analytical\nGaussian')
cur_ax.set_xlim((xmin,xmax))
cur_ax.set_ylim((0,ymax))
cur_ax.set_xlabel('Selection coefficient estimate, $\\^s$',fontsize=FontSize)
cur_ax.axes.get_yaxis().set_ticklabels([])
cur_ax.tick_params(axis='x', labelsize=FontSize)
cur_ax.tick_params(axis='y', labelsize=FontSize)
cur_ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
cur_ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))
cur_ax.legend(fontsize=FontSize,frameon=False,loc=2)

cur_ax.text(0.95,0.85,'$n_s = $' + str(ns_ref),ha='right',va='center',fontsize=FontSize, transform=cur_ax.transAxes)





axes[0].text(-0.08,1.15,'A',fontsize=18,transform=axes[0].transAxes,fontweight='bold',va='top',ha='right')
axes[1].text(-0.14,1.15,'B',fontsize=18,transform=axes[1].transAxes,fontweight='bold',va='top',ha='right')


plt.tight_layout()

cur_ax = axes[1]
x_pos = cur_ax.get_position().intervalx
y_pos = cur_ax.get_position().intervaly
cur_ax.set_position([x_pos[0]+0.03, y_pos[0], x_pos[1]-x_pos[0], y_pos[1]-y_pos[0]])


if save_flag:
    plt.savefig('./Figures/Fig4_AsympGauss.pdf',dpi=dpi_val,bbox_inches='tight')