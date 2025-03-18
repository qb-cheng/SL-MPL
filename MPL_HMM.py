# -*- coding: utf-8 -*-
"""
This file estimates selection coefficients from observed mutant allele frequency
 trajectories under the joint effect of limited sampling and genetic drift,
using MPL-based estimator and the extended one (effectively HMM).
"""

import Functions,GetCaseArg,pickle,time,os,sys
import numpy as np
from cpuinfo import get_cpu_info


save_flag = True

dt = 10
TrajNum = 1000 # Estimate selection coefficient from this number of observed trajectories

# Settings for the HMM
D = 100
TrainingConvergence = 1e-4


# Sets = [51,61,71] #1,11,21,31,41,51,61
# Ts = [50,100,150,300,450] #50,100,150,300,450
# nss = [100,200,500,1000] #10,50,100,200  # At max 250

# Sets = [11,21,31,41] #1,11,21,31,41,51,61
# Ts = [50,100,150,300,450] #50,100,150,300,450
# nss = [10,20,50,100] #10,50,100,200

Sets = [1,11,21,31,41,51,61,71] 
Ts = [20,50,100,150,300,450]
ns_ratios = [0.01,0.02,0.05,0.1,0.2]

for ns_ratio in ns_ratios:
    
    if save_flag:
        preDir = './ModelComp/ns_ratio_'+str(ns_ratio)
        if not os.path.exists(preDir):
            os.makedirs(preDir)
    
    
    for thisSet in Sets:
        N,u,s,x0,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
        
        with open('./PopRecords/Set' + str(thisSet) + '.pkl','rb') as f:
            StoTrajs = pickle.load(f)
          
         
        t = np.arange(0,T,dt)   
        ns = int(N * ns_ratio)
        ObservedTrajs = Functions.SampleOnce(StoTrajs[t,:], ns)
    
        
        for curT in Ts:
            FileToWrite = preDir+'/N'+str(N)+'_T'+str(curT)+'_HMM.pkl'
            if os.path.exists(FileToWrite):
                continue
            
            ests_sFS = np.zeros(TrajNum)
            ests_HMM = np.zeros(TrajNum)
        
            end_idx = int(curT / dt) + 1
            CurObservedTrajs = ObservedTrajs[:end_idx,:]
            
            # Running HMM estimator
            TimeCost_HMM = 0
            ValidCounter = 0
            ValidTrajIndices = [] # When N = 100, some sampled trajectories cannot be estimated properly with this estimator due to their ill conditions. Skip those trajectories
            
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            
            for Traj_idx in range(CurObservedTrajs.shape[1]):
                print(Traj_idx)
                CurObservedTraj = CurObservedTrajs[:,Traj_idx]
                try:
                    StartTime = time.time()
                    ests_HMM[ValidCounter] = Functions.MPL_HMM(CurObservedTraj,N,ns,D,x0,dt,u,TrainingConvergence)
                    TimeCost_HMM += time.time() - StartTime
                    ValidTrajIndices.append(Traj_idx)
                    ValidCounter += 1
                    
                    if ValidCounter == TrajNum:
                        break
                    
                except:
                    print(Traj_idx,"Skipped a problematic trajectory")
                    
            TimeCost_HMM /= TrajNum
            sys.stderr = original_stderr
            print(N,ns,curT,"done")
            
            # Running sFS estimator
            StartTime = time.time()
            # D_hat = CurSampledTrajs[-1,:] - CurSampledTrajs[0,:] - u*dT*np.sum(1-2*CurSampledTrajs[:-1,:],axis=0)
            # V_hat = dT * np.sum(CurSampledTrajs[:-1,:] * (1 - CurSampledTrajs[:-1,:]),axis=0)/(1-1/ns)
            # ests_SLMPL = D_hat/V_hat
            for run_idx,Traj_idx in enumerate(ValidTrajIndices):
                CurObservedTraj = CurObservedTrajs[:,Traj_idx]
                D_hat = CurObservedTraj[-1] - CurObservedTraj[0] - u*dt*np.sum(1-2*CurObservedTraj[:-1])
                V_hat = dt * np.sum(CurObservedTraj[:-1] * (1 - CurObservedTraj[:-1]))/(1-1/ns)
                ests_sFS[run_idx] = D_hat/V_hat
                
            TimeCost_sFS = time.time() - StartTime
            TimeCost_sFS /= TrajNum
            
            
            
            if save_flag:
                info = get_cpu_info()
                with open(FileToWrite,'wb') as f:
                # with open(preDir+'/N'+str(N)+'_T'+str(curT)+'_OptimizerTNC.pkl','wb') as f:
                    pickle.dump([ests_sFS,TimeCost_sFS,ests_HMM,TimeCost_HMM,info],f)