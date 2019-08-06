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
    S = P.shape[0]
    A = P.shape[1]
    P_pi = np.zeros((S, S))
    R_pi = np.zeros((S))
    for i in range(S):  # current state
        for a in range(A):
            for j in range(S): # next state
                # Explanation: P(s'|s) = sum_a pi(a|s)P(s'|s,a)
                P_pi[i, j] += pi[i, a] * P[i,a,j]
                is_row_updated = True
            R_pi[i] += pi[i, a] * R[i,a]

    if np.any(np.abs(P_pi.sum(axis=1) - 1) > 1e-5):
        raise RuntimeError('Probabilty matrix not normalized!!')
    return P_pi, R_pi

#------------------------------------------------------------------------------------------------------------~
def PolicyEvaluation(R, P, pi, gamma, P_pi=None, R_pi=None):
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
    S = P.shape[0]
    A = P.shape[1]
    if P_pi is None or R_pi is None:
        P_pi, R_pi = GetPolicyDynamics(R, P, pi)
    V_pi = np.linalg.solve((np.eye(S) - gamma * P_pi), R_pi)
    # Verify that R_pi + gamma * np.matmul(P_pi,  V_pi) == V_pi
    Q_pi = np.zeros((S, A))
    for a in range(A):
        for i in range(S):
            Q_pi[i, a] = R[i, a] + gamma * np.matmul(P[i,a,:], V_pi)
    # Verify that V_pi(s) = sum_a pi(a|s) * Q_pi(s,a)
    return V_pi, Q_pi

#------------------------------------------------------------------------------------------------------------~
def PolicyIteration(P, R, gamma):
    """
       Finds the optimal policy given a known model using policy-iteration algorithm

       Parameters:
       P: [S x A x S] transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)
       R: [S x A] mean rewards matrix R
       gamma: Discount factor

       Returns
       pi_opt [S x A]: An optimal policy (assuming given model and gamma)
       V_opt: [S] The optimal value-function , i,e. the the expected discounted return when following optimal policy  starting from some state
       Q_opt [S x A] The optimal Q-function, i,e. the the expected discounted return when following optimal policy starting from some state and action
       """

    # The algorithm: until policy not changes: (1) run policy-evaluation to get Q_pi  (2) new_policy = argmax Q
    S = P.shape[0]
    A = P.shape[1]
    Q_pi = np.zeros((S, A))
    # initial point of the algorithm: uniform policy
    pi = np.ones((S, A)) / A
    pi_prev = A - pi # arbitrary different policy than pi
    max_iter = S*A
    iter = 0
    while np.any(pi != pi_prev):
        pi_prev = pi
        _, Q_pi = PolicyEvaluation(R, P, pi, gamma)
        # Policy improvment:
        pi = np.zeros((S, A))
        pi[np.arange(S), np.argmax(Q_pi, axis=1)] = 1  #  set 1 for the optimal action w.r.t Q, and 0 for the other actions
        if iter > max_iter:
            raise RuntimeError('Policy Iteration should have stopped by now!')
        iter += 1

    pi_opt = pi
    Q_opt = Q_pi
    V_opt = np.max(Q_opt, axis=1)
    return pi_opt,  V_opt, Q_opt