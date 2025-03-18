# -*- coding: utf-8 -*-
"""
Trajectories are simulated with Wright-Fisher model based on the settings provided in GetCaseArg.py
Process involved: Selection, mutation
"""

import os,pickle,Functions
import numpy as np
import GetCaseArg


if __name__ == '__main__':
        
    PopulationDir = "./PopRecords"
    if not os.path.isdir(PopulationDir):
        os.mkdir(PopulationDir)
    
    # Sets = np.array([0,1,2,10,11,12,20,21,22,30,31,32,40,41,42])
    Sets = np.arange(601,631)
    
    for thisSet in Sets:
        print("EvoGen: Set" + str(thisSet) + " -->")
        # Step 1: Initialize and store basic parameters
        N,u,s,InitialAlleleFreq,NumItr,T = GetCaseArg.GetCaseInfo(thisSet)
        
            
        
        PopTraj = np.zeros((T,NumItr))
        for ItrIdx in range(NumItr):
            # Initialize population record data structure
            if type(InitialAlleleFreq) == list:
                x0 = np.round(np.random.uniform(InitialAlleleFreq[0],InitialAlleleFreq[1])*N)/N
            else:
                if InitialAlleleFreq > 1 or InitialAlleleFreq < 0:
                    x0 = np.random.randint(0,N+1)/N
                else:
                    x0 = InitialAlleleFreq
                
            PopTraj[:,ItrIdx] = Functions.WF_evolve(s,N,u,x0,T)
                    
            if ItrIdx * 100 % NumItr == 0:
                print(ItrIdx)
                
        
        # Save population records          
        with open(PopulationDir + "/Set" + str(thisSet) + ".pkl", "wb") as f:
            pickle.dump(PopTraj,f)
        
        
