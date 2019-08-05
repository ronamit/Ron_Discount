from tabular.data_utils import GenerateMDP, SampleTrajectories
from tabular.planing_utils import GetUniformPolicy, GetPolicyDynamics, PolicyEvaluation

###############  main   ##################
# * M := GenerateMDP
N = 5  # number of states
A = 2  # number of actions
k = 2  #  Number of non-zero entries in each row  of transition-matrix

P, R = GenerateMDP(N, A, k)

# * Set a fixed exploration policy pi  -  uniform over actions
pi = GetUniformPolicy(N, A)

P_pi, R_pi = GetPolicyDynamics(R, P, pi)
gammaEval = 0.95

# The true value-function:
V_pi, Q_pi = PolicyEvaluation(R, P, pi, gammaEval, P_pi, R_pi)

n = 2  #  number of trajectories to generate
depth = 10  # Length of trajectory
data = SampleTrajectories(P, R, pi, n, depth, p0=None, reward_std=0.1)


# * Use 2 methods for estimation:
#   1. Model-based - estimate M_hat from D and do PolicyEvaluation
#   2. Model-free TD-learning  using D


print('done')
