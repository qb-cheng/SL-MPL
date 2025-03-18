# -*- coding: utf-8 -*-


import numpy as np
import scipy, warnings
import scipy.stats as st
from hmmlearn import hmm
from scipy.stats import binom
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore") 
# import statsmodels.api as sm

def DeterministicTrajectory(s,x0,T,u):
    """
    Compute the deterministic trajectory
    Input arguments:
        s:  selection coefficient
        x0: initial allele frequency
        T:  number of generations (i.e., trajectory length)
        u:  mutation rate 
    """
    DeterministicTraj = np.zeros(T)
    DeterministicTraj[0] = x0
    for t in range(1,T):
        DeterministicTraj[t] = (u*(1-DeterministicTraj[t-1]) + (1+s)*(1-u)*DeterministicTraj[t-1]) / (1+s*DeterministicTraj[t-1])

    return DeterministicTraj


def WF_evolve(s,N,u,x0,T):
    """
    Use Wright-Fisher model to simulate one trajectory
    Input arguments:
        s:  selection coefficient
        N:  population size
        u:  mutation rate
        x0: initial allele frequency
        T:  number of generations (i.e., trajectory length)
    """
    PopTraj = np.zeros(T)
    PopTraj[0] = x0
    for t in range(1,T):
        MutantNum = int(PopTraj[t-1] * N) # Number of mutant alleles in the population
        CurrentPop = np.concatenate((np.zeros(N-MutantNum,int),np.ones(MutantNum,int))) # Construct the population
        
        # Mutation
        if u > 0:
            MutationSwitch = (np.random.rand(N) < u).astype(int)
            CurrentPop = np.abs(CurrentPop - MutationSwitch)
        
        # Selection
        CurFreq = np.mean(CurrentPop)
        p = ((1+s) * CurFreq) / (1 + s*CurFreq) # WF probability
        
        # Record frequency after WF process
        PopTraj[t] = np.random.binomial(N,p) / N
        
    return PopTraj


def SampleTrajectory(PopTraj,ns,MCRun):
    """
    Sample the given trajectory with binomial.
    Note that this is BINOMIAL sampling, which is with replacement.
    It it NOT equivalent to taking alleles from the population, unless N->inf
    Input arguments: 
        PopTraj:    the given trajectory
        ns:         constant sample size, number of accessible sequences
        MCRun:      number of Monte-Carlo run used to compute statistics
    """
    Sampledtraj = np.zeros((len(PopTraj),MCRun))
    for idx in range(len(PopTraj)):
        if type(ns) == np.ndarray:
            Sampledtraj[idx,:] = np.random.binomial(ns[idx],PopTraj[idx],MCRun)/ns[idx]
        else:
           Sampledtraj[idx,:] = np.random.binomial(ns,PopTraj[idx],MCRun)/ns
           
    return Sampledtraj


def SampleOnce(PopTrajs,ns):
    """
    Sample each trajectory once.
    This function adds sampling noise to the given trajectories.
    Note that this is BINOMIAL sampling, with replacement.
    Input arguments: 
        PopTraj:    the given trajectory
        ns:         constant sample size, number of accessible sequences
    """
    Sampledtrajs = np.zeros(PopTrajs.shape)
    
    if PopTrajs.ndim == 1:
        for r_idx in range(PopTrajs.shape[0]):
            Sampledtrajs[r_idx] = np.random.binomial(ns, PopTrajs[r_idx]) / ns
    elif PopTrajs.ndim == 2:
        for r_idx in range(PopTrajs.shape[0]):
            for c_idx in range(PopTrajs.shape[1]):
                Sampledtrajs[r_idx,c_idx] = np.random.binomial(ns, PopTrajs[r_idx,c_idx]) / ns  
    
    return Sampledtrajs


def ComputeIV(ObservableTraj,dt):
    """
    Compute the integrated allele variance of a frequency trajectory
    Input arguments:
        Traj:   the observable mutant allele frequency trajectory
        dt:     time sampling step
    Note that the first element in the returned array is when the trajectory length T = 2*dt
    """
    IV = np.cumsum(ObservableTraj * (1-ObservableTraj)) * dt
    return IV[1:-1]



def TaylorTerms(ObservableTraj,dt,ns,u):
    """
    Compute the three terms appeared in the approxiamted estimation varirance Var[\^s_{FS}]
    Input arguments:
        ObservableTraj: the given population mutant allele frequency trajectory, at observable generations
        dt:             time sampling step
        ns:             constant sample size
        u:              mutation rate
    Note that the first element in the returned array is when the trajectory length T = 2*dt
    """
    
    ED = (ObservableTraj[1:]-ObservableTraj[0] - u*dt*np.cumsum(1-2*ObservableTraj[:-1]))
    
    AlleleVar = ObservableTraj * (1-ObservableTraj)
    EV = np.cumsum(dt*(AlleleVar[:-1])) # the population integrated variance
    
    
    VarD = 1/ns * (AlleleVar[1:] + (1-4*u*dt)*AlleleVar[0] + 4*u**2*dt*EV)
    CovDV = -dt/ns * (AlleleVar[0]*(1-2*ObservableTraj[0]) - 2*u*dt*np.cumsum(AlleleVar[:-1]*(1-2*ObservableTraj[:-1])))
    VarV = dt/ns * (EV - (4-2/(ns-1))*dt*np.cumsum(AlleleVar[:-1]**2))
    
    return ED[1:],EV[1:],VarD[1:],CovDV[1:],VarV[1:]



def TaylorTerms_general(ObservableTraj,dates,nss,u):
    """
    Compute the three terms appeared in the approxiamted estimation varirance Var[\^s_{FS}]
    Input arguments:
        AccessibleTraj:   the given accessible trajectory
        dates:            the recorded dates
        nss:               number of recorded sequences at each time point
        u:                mutation rate
    Note that the first element in the returned array is when the trajectory length T = 2*dt
    """
    AlleleVar = ObservableTraj * (1-ObservableTraj)
    dts = dates[1:] - dates[:-1]
    
    ED = (ObservableTraj[1:]-ObservableTraj[0] - u*np.cumsum(dts*(1-2*ObservableTraj[:-1])))
    EV = np.cumsum(dts*(AlleleVar[:-1])) # the true integrated variance, without gamma
    
    
    VarD = (AlleleVar[1:]/nss[1:] + (1-4*u*dts[0])*AlleleVar[0]/nss[0] + 4*u**2*np.cumsum(dts**2*AlleleVar[:-1]/nss[:-1]))
    VarV = np.cumsum(dts**2*(AlleleVar[:-1])/nss[:-1]) - np.cumsum(dts**2*(AlleleVar[:-1]**2)*(4-2/(nss[:-1]-1))/nss[:-1])
    CovDV = - dts[0] * AlleleVar[0]*(1-2*ObservableTraj[0]) /nss[0] + 2*u * np.cumsum(dts**2*AlleleVar[:-1]*(1-2*ObservableTraj[:-1])/nss[:-1])
    return ED[1:],EV[1:],VarD[1:],CovDV[1:],VarV[1:]




def SLMPL(ObservedTraj,dt,ns,u):
    """
    Estimate selection coefficient from frequency trajectories, at observable generations
    If ns > 0: the trajectory is obtained from observations under finite sampling effects
    Otherwise: the trajectory represents the population, without finite sampling effects
    Input arguments:
        Traj:   the observed mutant allele frequency trajectory
        dt:     time sampling step
        ns:     sampling size, number of accessible sequences
        u:      mutation rate
    Note that the first element in the returned array is when the trajectory length T = 2*dt
    """
    if len(ObservedTraj.shape) != 1 and len(ObservedTraj.shape) != 2:
        # Input is not valid
        print("Invalid input. Trajectory should be 1D or 2D.")
        return 100
        
    if len(ObservedTraj.shape) == 1:
        # Input is a one-dimensional trajectory
        
        D = ObservedTraj[1:] - ObservedTraj[0] - u*dt*np.cumsum(1-2*ObservedTraj[:-1])
        V = dt*np.cumsum(ObservedTraj[:-1]*(1-ObservedTraj[:-1]))
    else:
        # Input is multiple trajectories        
        D = ObservedTraj[1:,:] - ObservedTraj[0,:] - u*dt*np.cumsum(1-2*ObservedTraj[:-1,:],axis=0)
        V = dt*np.cumsum(ObservedTraj[:-1,:]*(1-ObservedTraj[:-1,:]),axis=0)
        
    if ns > 0:
        # The input is finite sampling observations, need to correct the bias
        V = V / (1-1/ns)
    
    estimates = D / V
    return estimates[1:]






def KL_div(estimates,resolution):
    """
    Normalize estimates to zero-mean and unit-variance, then compute the KL divergence between normal distribution N(0,1)
    estimates : sample estimates \^s_FS
    resolution : resolution on the domain (discretized pdf)
    """
    estimates_normed = (estimates - np.mean(estimates)) / np.std(estimates)
    bin_low = np.floor(np.min(estimates_normed)/resolution)*resolution
    bin_high = np.ceil(np.max(estimates_normed)/resolution)*resolution
    bins = np.arange(bin_low,bin_high+resolution,resolution)

    estimates_digitized = np.digitize(estimates_normed, bins)
    NonEmptyBinNum,BinCount = np.unique(estimates_digitized,return_counts=True)

    p = np.zeros(len(bins)-1,int)
    for idx in range(len(bins)-1):
        if idx+1 not in NonEmptyBinNum:
            continue
        p[idx] = BinCount[np.where(NonEmptyBinNum == idx+1)[0][0]]
        
    p = p / np.sum(p)
    bins_mid = (bins[:-1] + bins[1:]) / 2
    q = scipy.stats.norm.pdf(bins_mid,0,1) * resolution

    kld = 0
    for idx,cur_p in enumerate(p):
        if cur_p == 0:
            continue
        kld += cur_p * np.log(cur_p / q[idx])
    
    return kld




def SLMPL_TimeVaryingN(ObservedTraj,dt,ns,u,N):
    """
    Similar to SLMPL, but this time with time-varying N
    The input N should be an array that records the population size at each observed generation
    """
    N = np.atleast_2d(N).T
    
    if len(ObservedTraj.shape) != 1 and len(ObservedTraj.shape) != 2:
        # Input is not valid
        print("Invalid input. Trajectory should be 1D or 2D.")
        return 100
        
    if len(ObservedTraj.shape) == 1:
        # Input is a one-dimensional trajectory
        D = np.cumsum(N[:-1] * (ObservedTraj[1:] - ObservedTraj[:-1] - u*dt*(1 - 2*ObservedTraj[:-1])))
        V = dt*np.cumsum(N[:-1] * ObservedTraj[:-1]*(1-ObservedTraj[:-1]))
    else:
        # Input contains multiple trajectories        
        D = np.cumsum(N[:-1] * (ObservedTraj[1:,:] - ObservedTraj[:-1,:] - u*dt*(1 - 2*ObservedTraj[:-1,:])), axis=0)
        V = dt*np.cumsum(N[:-1] * ObservedTraj[:-1,:]*(1-ObservedTraj[:-1,:]),axis=0)
        
    if ns > 0:
        # The input is finite sampling observations, need to correct the bias
        V = V / (1-1/ns)
    
    estimates = D / V
    return estimates[1:]



def LL_observation(s,ObservedDeterTraj,dt,ns,u,x0):
    # Log-likelihood of the observation given the determinisic population frequency trajectory
    # return the negative of log-likelihood, and put it into a (minimization) optimizer
    TrajLen = (len(ObservedDeterTraj) - 1) * dt + 1
    
    DeterTraj_tentative = np.zeros(TrajLen)
    DeterTraj_tentative[0] = x0
    for t_idx in range(1,TrajLen):
        DeterTraj_tentative[t_idx] = (u*(1-DeterTraj_tentative[t_idx-1]) + (1+s)*(1-u)*DeterTraj_tentative[t_idx-1]) / (1+s*DeterTraj_tentative[t_idx-1])
    
    LL = 0
    for t_idx,cur_t in enumerate(np.arange(dt,TrajLen,dt)):
        # Do not compute the likelihood of observed x0 as it is assumed to be known
        LL += np.log(st.binom.pmf(int(ObservedDeterTraj[t_idx+1]*ns),ns,DeterTraj_tentative[cur_t]))
    
    return -LL


def MPL_HMM(CurObservedTraj,N,ns,D,x0,dt,u,TrainingConvergence):
    """
    The HMM-based selection coefficient estimator
    CurObservedTraj: the Traj at observed generations, not necessarily the complete one
    N: population size
    ns: sample size
    D: number of grids to uniformly separate frequency space to intervals
    x0: initial allele frequency
    dt: time sampling step
    u: mutation rate
    TrainingConvergence: threshold to stop iteration when estimate change is smaller
    """
    
    dx = 1/D
    x = np.linspace(dx/2,1-dx/2,D)
        
    start_prob = np.zeros(len(x))
    start_prob[next(idx for idx in range(len(x)) if x[idx] >= x0)] = 1 # Initial mutant allele frequency assumed to be known, i.e., start_prob is a delta function
    # lower <= x < upper (left inclusive)

    ObservedCounts = (CurObservedTraj*ns).astype(int)


    def LogLikelihood(SC):
        # Return the negative log likelihood to be minimized (i.e., therefore maximizing likelihood)
        # Set up the HMM
        trans_prob = np.zeros((D,D))
        for row_idx in range(D):
            for col_idx in range(D):
                CurFreq = (row_idx+0.5)/D
                CurAlleleVar = CurFreq * (1-CurFreq)
                NextFreq = (col_idx+0.5)/D
                
                # Diffusion approximation + path integral
                drift = SC*CurAlleleVar + u*(1-2*CurFreq)
                trans_prob[row_idx,col_idx] = np.sqrt(N/(2*np.pi*dt)) * dx/np.sqrt(CurAlleleVar) * np.exp(-N/(2*dt) * (NextFreq - CurFreq - dt*drift)**2/CurAlleleVar)

                # Pr(r->c)
                
            # Normalise it to ensure all probabilities sum to 1
            trans_prob[row_idx,:] = trans_prob[row_idx,:] / np.sum(trans_prob[row_idx,:])
        
        
        
        
        emiss_prob = np.zeros((D,2))
        for row_idx in range(D):
            CurFreq = (row_idx+0.5)/D
            emiss_prob[row_idx,0] = 1 - CurFreq
            emiss_prob[row_idx,1] = CurFreq
            
        
        Observations = np.zeros((len(ObservedCounts),2),int)
        Observations[:,0] = ns - ObservedCounts
        Observations[:,1] = ObservedCounts
        
        
        
        # emiss_prob = np.zeros((D,ns+1))
        # for row_idx in range(D):
        #     for col_idx in range(ns+1):
        #         CurFreq = (row_idx+0.5)/D
        #         # ObservedFreq = col_idx/ns
                
        #         emiss_prob[row_idx,col_idx] = scipy.stats.binom.pmf(col_idx,ns,CurFreq)
        #         # Pr(state->observation)
                
        #     emiss_prob[row_idx,:] = emiss_prob[row_idx,:]/np.sum(emiss_prob[row_idx,:])
                

        model = hmm.MultinomialHMM(n_components=D,n_trials=ns,init_params='') # n_iter=10
        model.n_features = 2
        model.startprob_ = start_prob
        model.transmat_ = trans_prob
        model.emissionprob_ = emiss_prob
        
        
        return -model.score(Observations,len(CurObservedTraj))
        
        # return -model.score(ObservedCounts)
    
    
    res = minimize(LogLikelihood, 0.02, method='Nelder-Mead', tol=TrainingConvergence)
    # res = scipy.optimize.minimize(LogLikelihood, 0.02, method='TNC', tol=TrainingConvergence)

    return res.x[0]





def Customized_MPL_HMM(ObservedCounts,N,ns_array,D,x0,ObGens,u,TrainingConvergence):
    """
    The HMM-based selection coefficient estimator
    ObservedCounts: array of the number of observed mutant alleles
    N: population size
    ns_array: array of sample sizes at each observed time point
    D: number of grids to uniformly separate frequency space to intervals
    x0: initial allele frequency
    ObGens: generation indices of observed time points
    u: mutation rate
    TrainingConvergence: threshold to stop iteration when estimate change is smaller
    """
    
    dx = 1/D
    x = np.linspace(dx/2,1-dx/2,D)
        
    start_prob = np.zeros(len(x))
    start_prob[next(idx for idx in range(len(x)) if x[idx] >= x0)] = 1 # Initial mutant allele frequency assumed to be known, i.e., start_prob is a delta function
    # lower <= x < upper (left inclusive)
    
    # start_prob = np.ones(len(x)) / len(x)
    
    def NegLogLikelihood(SC):
        # Return the negative log likelihood to be minimized (i.e., therefore maximizing likelihood)
        
        emiss_prob = np.zeros(D)
        for row_idx in range(D):
            CurFreq = (row_idx+0.5)/D
            emiss_prob[row_idx] = CurFreq
        
        T = len(ObservedCounts)
        alpha = np.zeros((T, D))
        
        # Initial step
        for i in range(D):
            alpha[0][i] = start_prob[i] * binom.pmf(ObservedCounts[0], ns_array[0], emiss_prob[i])
        
        TransProbs = {}
        # Recursive step
        for t_idx in range(1, T):
            cur_dt = ObGens[t_idx] - ObGens[t_idx-1]
            if cur_dt in TransProbs:
                cur_trans_prob = TransProbs[cur_dt]
            else:
                cur_trans_prob = np.zeros((D,D))
                for row_idx in range(D):
                    for col_idx in range(D):
                        CurFreq = (row_idx+0.5)/D
                        CurAlleleVar = CurFreq * (1-CurFreq)
                        NextFreq = (col_idx+0.5)/D
                        
                        # Diffusion approximation + path integral
                        drift = SC*CurAlleleVar + u*(1-2*CurFreq)
                        cur_trans_prob[row_idx,col_idx] = np.sqrt(N/(2*np.pi*cur_dt)) * dx/np.sqrt(CurAlleleVar) * np.exp(-N/(2*cur_dt) * (NextFreq - CurFreq - cur_dt*drift)**2/CurAlleleVar)
                        
                    # Normalise it to ensure all probabilities sum to 1
                    cur_trans_prob[row_idx,:] = cur_trans_prob[row_idx,:] / np.sum(cur_trans_prob[row_idx,:])
                TransProbs[cur_dt] = cur_trans_prob
                
            for j in range(D):
                sum_prev_alpha = sum(alpha[t_idx-1][i] * cur_trans_prob[i][j] for i in range(D))
                alpha[t_idx][j] = sum_prev_alpha * binom.pmf(ObservedCounts[t_idx], ns_array[t_idx], emiss_prob[j])
    
        # Termination
        likelihood = sum(alpha[T-1][i] for i in range(D))       
        return -np.log(likelihood)
        
    
    
    res = minimize(NegLogLikelihood, 0.02, method='Nelder-Mead', tol=TrainingConvergence)
    mle = res.x[0]
    neg_log_likelihood_mle = NegLogLikelihood(mle)

    # Define the likelihood ratio function
    def likelihood_ratio(SC):
        return 2 * (NegLogLikelihood(SC) - neg_log_likelihood_mle)

    # Find the confidence interval using the chi-square distribution
    chi2_critical_value = 3.84  # 95% confidence interval for 1 degree of freedom

    # Search for the lower bound
    ci_lower = minimize_scalar(
        lambda SC: np.abs(likelihood_ratio(SC) - chi2_critical_value),
        bounds=(-1, mle),
        method='bounded'
    ).x

    # Search for the upper bound
    ci_upper = minimize_scalar(
        lambda SC: np.abs(likelihood_ratio(SC) - chi2_critical_value),
        bounds=(mle, 1),
        method='bounded'
    ).x

    return mle, (ci_lower, ci_upper)