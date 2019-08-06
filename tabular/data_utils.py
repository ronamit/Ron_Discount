from __future__ import division, absolute_import, print_function
import numpy as np

#------------------------------------------------------------------------------------------------------------~
def GenerateMDP(S, A, k):
    """
    Randomly generate an MDP from the Random-MDP distribution.

    For each state-action pair (s; a), the distribution over the next state,  P_{s,a,s'}=P(s'|s,a), is determined by choosing k  non-zero entries uniformly from
     all S states, filling these k entries with values uniformly drawn from [0; 1], and finally normalizing


    Parameters:
    S: number of states
    A: number of actions
    k: Number of non-zero entries in each row  of transition-matrix

    Returns:
    P: [S x A x S] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R: [S x A] mean rewards matrix R
    """
    P = np.zeros((S, A, S))
    for a in range(A):
        for i in range(S):
            nonzero_idx = np.random.choice(S, k, replace=False)
            for j in nonzero_idx:
                P[i, a, j] = np.random.rand(1)
            P[i, a, :] /= P[i, a, :].sum()
    R = np.random.rand(S, A)
    return P, R

#------------------------------------------------------------------------------------------------------------~
def SampleTrajectories(P, R, pi, n, depth, p0=None, reward_std=0.1):
    """
    # generate n trajectories


    Parameters:
    P: [S x A x S] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R: [S x A] mean rewards matrix R
    pi: [S x A]  matrix representing  pi(a|s)
    n: number of trajectories to generate
    depth: Length of trajectory
    p0 (optional) [S] matrix of initial state distribution (default:  uniform)
    Returns:
    data: list of n trajectories, each is a list of sequence of depth tuples (state, action, reward, next state)
    """
    S = P.shape[0]
    A = P.shape[1]
    if p0 is None:
        p0 = np.ones(S) / S #  uniform
    data = []
    for i_traj in range(n):
        data.append([])
        # sample initial state:
        s = np.random.choice(S, size=1, p=p0)[0]
        for t in range(depth):
            # Until t==depth, sample a~pi(.|s), s'~P(.|s,a), r~R(s,a)
            a = np.random.choice(A, size=1, p=pi[s, :])[0]
            s_next = np.random.choice(S, size=1, p=P[s, a, :])[0]
            r = R[s,a] + np.random.randn(1)[0] * reward_std
            data[i_traj].append((s,a,r,s_next))
            s = s_next
    return data