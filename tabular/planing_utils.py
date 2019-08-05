from __future__ import division, absolute_import, print_function
import numpy as np

#------------------------------------------------------------------------------------------------------------~
def GetUniformPolicy(N, A):
    """
    Create a Markov stochastic policy which chooses actions randomly uniform from each state

    Parameters:
    N: number of states
    A: number of actions

    Returns:
    pi: [S x A]  matrix representing  pi(a|s)
    """
    pi = np.ones((N, A))
    for i in range(N):
        pi[i] /= pi[i].sum()
    return pi

#------------------------------------------------------------------------------------------------------------~
def GetPolicyDynamics(R, P, pi):
    """
    Calculate the dynamics when following the policy pi

    Parameters:
    P: [S x A x S] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R: [S x A] mean rewards matrix R
    pi: [S x A]  matrix representing  pi(a|s)

    Returns:
    P_pi: [S x S] transitions matrix  when following the policy pi      (P_pi)_{s,s'} P^pi(s'|s)
    R_pi: [S] mean rewards at each state when following the policy pi    (R_pi)_{s} = R^pi(s)
    """
    N = P.shape[0]
    A = P.shape[1]
    P_pi = np.zeros((N, N))
    R_pi = np.zeros((N))
    for i in range(N): # current state
        for a in range(A):
            for j in range(N): # next state
                # Explanation: P(s'|s) = sum_a pi(a|s)P(s'|s,a)
                P_pi[i, j] += pi[i, a] * P[i,a,j]
            R_pi[i] += pi[i, a] * R[i,a]
    return P_pi, R_pi

#------------------------------------------------------------------------------------------------------------~
def PolicyEvaluation(R, P, pi, gamma):
    """
    Calculates the value-function for a given policy pi and a known model

    Parameters:
    P: [S x A x S] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
    R: [S x A] mean rewards matrix R
    pi: [S x A]  matrix representing  pi(a|s)
    gamma: Discount factor

    Returns:
    V_pi: [S] The value-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state
    Q_pi [S x A] The Q-function for a fixed policy pi, i,e. the the expected discounted return when following pi starting from some state and action
    """
    # (1) Use PolicyDynamics to get P and R, (2) V = (I-gamma*P)^-1 * R
    N = P.shape[0]
    P_pi, R_pi = GetPolicyDynamics(R, P, pi)
    V_pi = np.linalg.solve((np.eye(N) - gamma * P_pi).T, P_pi.T).T
    Q_pi = V_pi
    return V_pi, Q_pi

