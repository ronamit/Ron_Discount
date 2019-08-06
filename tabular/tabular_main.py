from __future__ import division, absolute_import, print_function
import numpy as np
from tabular.data_utils import GenerateMDP, SampleTrajectories
from tabular.planing_utils import GetUniformPolicy, GetPolicyDynamics, PolicyEvaluation, PolicyIteration
from tabular.learning_utils import ModelEstimation

###############  Create Data   ##################
# * M := GenerateMDP
S = 10  # number of states
A = 2  # number of actions
k = 5  #  Number of non-zero entries in each row  of transition-matrix

P, R = GenerateMDP(S, A, k)

# * Set a fixed exploration policy pi  -  uniform over actions
pi = GetUniformPolicy(S, A)

P_pi, R_pi = GetPolicyDynamics(R, P, pi)
gammaEval = 0.99

# The true value-function:
V_pi, Q_pi = PolicyEvaluation(R, P, pi, gammaEval, P_pi, R_pi)

n = 2 #  number of trajectories to generate
depth = 10  # Length of trajectory
data = SampleTrajectories(P, R, pi, n, depth, p0=None, reward_std=0.1)

##############  Policy Evaluation   ##################
# * Use 2 methods for estimation:
#   1. Model-based - estimate M_hat from D and do PolicyEvaluation
#   2. Model-free TD-learning  using D

P_est, R_est = ModelEstimation(data, S, A)

#
for gamma_guidance in  [0.98, 0.985, 0.99]:
    V_est, Q_est = PolicyEvaluation(R_est, P_est, pi, gamma_guidance)


    bias_err = (V_pi - V_est).mean()
    estErr = np.sqrt(np. square(V_pi - V_est).mean())
    print('Policy-evaluation loss: ', estErr, '  gamma_guidance: ', gamma_guidance)
##############  Optimal Control (as in Jiang '15)   ##################

pi_opt,  V_opt, Q_opt = PolicyIteration(P, R, gammaEval)


# for reps
# generate data
# Estimate model
# for loop on gamma_guidance
for gamma_guidance in  [0.98, 0.985, 0.99]:
    # CE policy w.r.t model-estimation and gamma_guidance
    pi_CE,  _, _ = PolicyIteration(P_est, R_est, gamma_guidance)
    # Evaluate perfomance of CE policy:
    V_CE, _ = PolicyEvaluation(R, P, pi_CE, gammaEval)

    planLoss = np.sqrt(np. square(V_opt - V_CE).mean())
    print('Planing loss: ', planLoss, '  gamma_guidance: ', gamma_guidance)
    print('done')
