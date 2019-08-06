from __future__ import division, absolute_import, print_function
import numpy as np
from tabular.data_utils import GenerateMDP, SampleTrajectories
from tabular.planing_utils import GetUniformPolicy, GetPolicyDynamics, PolicyEvaluation, PolicyIteration
from tabular.learning_utils import ModelEstimation

###############  Model definition   ##################
S = 10  # number of states
A = 2  # number of actions
k = 5  #  Number of non-zero entries in each row  of transition-matrix
gammaEval = 0.99


##############  Policy Evaluation   ##################
# * Use 2 methods for estimation:
#   1. Model-based - estimate M_hat from D and do PolicyEvaluation
#   2. Model-free TD-learning  using D

P, R = GenerateMDP(S, A, k)

# * Set a fixed exploration policy pi  -  uniform over actions
pi = GetUniformPolicy(S, A)

P_pi, R_pi = GetPolicyDynamics(P, R, pi)

# The true value-function:
V_pi, Q_pi = PolicyEvaluation(P, R, pi, gammaEval, P_pi, R_pi)

n = 2 #  number of trajectories to generate
depth = 10  # Length of trajectory
data = SampleTrajectories(P, R, pi, n, depth, p0=None, reward_std=0.1)

P_est, R_est = ModelEstimation(data, S, A)

gamma_guidance = 0.9
V_est, Q_est = PolicyEvaluation(P_est, R_est, pi, gamma_guidance)

bias_err = (V_pi - V_est).mean()
estErr = np.sqrt(np. square(V_pi - V_est).mean())
print('Policy-evaluation loss: ', estErr, '  gamma_guidance: ', gamma_guidance)

print('done')
