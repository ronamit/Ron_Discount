from __future__ import division, absolute_import, print_function
import numpy as np

#------------------------------------------------------------------------------------------------------------~
def GenerateMDP(N, A, k):
    """
    Randomly generate an MDP from the Random-MDP distribution.

    For each state-action pair (s; a), the distribution over the next state,  P_{s,a,s'}=P(s'|s,a), is determined by choosing k  non-zero entries uniformly from
     all N states, filling these k entries with values uniformly drawn from [0; 1], and finally normalizing


    Parameters:
    N: number of states
    A: number of actions
    k: Number of non-zero entries in each row  of transition-matrix

    Returns:
    P: [S x A x S] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R: [S x A] mean rewards matrix R
    """
    P = np.zeros((N, A, N))
    for a in range(A):
        for i in range(N):
            nonzero_idx = np.random.choice(N, k, replace=False)
            for j in nonzero_idx:
                P[i, a, j] = np.random.rand(1)
            P[i, a, :] /= P[i, a, :].sum()
    R = np.random.rand(N, A)
    return P, R