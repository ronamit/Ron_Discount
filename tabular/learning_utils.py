from __future__ import division, absolute_import, print_function
import numpy as np

#------------------------------------------------------------------------------------------------------------~
def ModelEstimation(data, S, A):
    """
    Maximum-Likelihood estimation of model based on data

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    S: number of states
    A: number of actions

    Returns:
    P_est: [S x A x S] estimated transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R_est: [S x A] estimated mean rewards matrix R
    """

    counts_sas = np.zeros((S,A,S))
    counts_sa =  np.zeros((S,A))
    R_est = np.zeros((S,A))
    P_est = np.zeros((S,A,S))
    for traj in data:
        for sample in traj:
            (s,a,r,s_next) = sample
            counts_sa[s,a] += 1
            counts_sas[s,a,s_next] += 1
            R_est[s,a] += r

    for s in range(S):
        for a in range(A):
            if counts_sa[s,a] == 0:
                # if this state-action doesn't exist in data
                # Use default values:
                R_est[s,a] = 0.5
                P_est[s,a,:] = 1/S
            else:
                R_est[s,a] /= counts_sa[s,a]
                P_est[s, a, :] = counts_sas[s,a,:] / counts_sa[s,a]
    if np.any(np.abs(P_est.sum(axis=2) - 1) > 1e-5):
        raise RuntimeError('Probabilty matrix not normalized!!')
    return P_est, R_est



#------------------------------------------------------------------------------------------------------------~
def TD_policy_evaluation(data, S, A, gamma):
    """
    Maximum-Likelihood estimation of model based on data

    Parameters:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    S: number of states
    A: number of actions
    gamma: Discount factor

    Returns:
    Returns:
    V_pi: [S] The estimated value-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state
    Q_pi [S x A] The estimated Q-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """
    n_iter = 1000
    stop_diff = 1e-5
    V_pi = np.zeros((S))
    # Join list of data tuples from all trajectories:
    data_tuples = sum(data, [])
    n_samples = len(data_tuples)
    for i_iter in range(n_iter):
        # Choose random sample:
        alpha = 100 / (1000 + i_iter)
        i_sample = np.random.randint(n_samples)
        (s, a, r, s_next) = data_tuples[i_sample]
        delta = r + gamma * V_pi[s_next] - V_pi[s]
        V_pi[s] += alpha * delta
    return V_pi

