from tabular.data_utils import GenerateMDP
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
gamma = 0.95
V_pi, Q_pi = PolicyEvaluation(R, P, pi, gamma)

print('done')

# * FindOptimalPolicy (for M, gammaEval)
# * calculate  average over states true-value for optimal policy , with gammaEval(first term in eq. 14) (use PolicyEvaluation)

# * D := Sample N trajectories with depth T (use SampleTrajectories)
# *  USe PolicyEvaluation to get V^pi[M,gammaEval]
# * Use 2 methods for estimation:
#   1. Model-based - estimate M_hat from D and do PolicyEvaluation
#   2. Model-free TD-learning  using D

# Next phase:
# 1. Plot performance vs guidance-gamma for different N


###############  subroutines   ##################

#  subroutine:  GenerateMDP: Output P(s'|s,a), R(s,a) (see randMat.m)

# subroutine:  FindOptimalPolicy (see dp.m)
# Compute the optimal-policy using policy-iteration:
# until policy not changes: (1) run policy-evaluation to get Q_pi  (2) new_policy = argmax Q

# subroutine: PolicyDynamics Input: P(s'|s,a), R(s,a), pi, Output:L P^pi(s'|s), R^pi(s) (see pol PR.m)
# % P(s'|s) = sum_a pi(a|s)P(s'|s,a)

# subroutine: PolicyEvaluation (input: pi, model,  output:  Q_pi and V_pi)  (see evalPol.m)
# (1) Use PolicyDynamics to get P and R, (2) V = (I-gamma*P)^-1 * R


# subroutine: SampleTrajectories (see sampleTrajectories.m)
# For each trajectory, create sequence of T (state, action, reward, next - state) tuples
# As initial state distribution use uniform over states
# Until T or terminal state, sample a~pi(.|s), s'~P(.|s,a), r~R(s,a)

# subroutine: ModelEstimation  (see trajectory2model.m)
# Input: data-tuples, output:  ML-estimation of model R,P

