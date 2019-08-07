"""
In  tabular MDP setting, evaluates the learning of policy evaluation for fixed policy using different guidance discount factors
"""
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
from tabular.data_utils import GenerateMDP, SampleTrajectories
from tabular.planing_utils import GetUniformPolicy, GetPolicyDynamics, PolicyEvaluation, PolicyIteration
from tabular.learning_utils import ModelEstimation, TD_policy_evaluation

###############  Model definition   ##################
S = 10  # number of states
A = 2  # number of actions
k = 5  #  Number of non-zero entries in each row  of transition-matrix
reward_std = 0.1
gammaEval = 0.99

n = 5  # number of trajectories to generate
depth = 10  # Length of trajectory

n_reps = 100  # number of experiment repetitions


method = 'TD-Learning'  #  'Model-Based' | 'TD-Learning'
do_correction = True

##############  Policy Evaluation   ##################
# * Use 2 methods for estimation:
#   1. Model-based - estimate M_hat from D and do PolicyEvaluation
#   2. Model-free TD-learning  using D

gamma_grid = np.linspace(0.1, 0.99, num=100)
n_gammas = gamma_grid.shape[0]
evaluation_loss = np.zeros((n_gammas, n_reps))
train_loss = np.zeros((n_gammas, n_reps))
test_loss = np.zeros((n_gammas, n_reps))

for i_rep in range(n_reps):
    # Generate MDP:
    P, R = GenerateMDP(S, A, k)

    # Set a fixed exploration policy pi  -  uniform over actions - for gathering data and for evaluating
    pi = GetUniformPolicy(S, A)

    # The true policy dynamics model
    P_pi, R_pi = GetPolicyDynamics(P, R, pi)

    # The true value-function (w.r.t  true model and gammaEval):
    V_pi, Q_pi = PolicyEvaluation(P, R, pi, gammaEval, P_pi, R_pi)

    # Generate data:
    data = SampleTrajectories(P, R, pi, n, depth, p0=None, reward_std=reward_std)


    # Estimate model
    for i_gamma, gamma_guidance in enumerate(gamma_grid):

        if method == 'Model-Based':
            # Estimate model:
            P_est, R_est = ModelEstimation(data, S, A)

            # Estimated V_pi according to estimated model and gamma_guidance:
            V_est, Q_est = PolicyEvaluation(P_est, R_est, pi, gamma_guidance)

        elif method == 'TD-Learning':
            V_est = TD_policy_evaluation(data, S, A, gamma_guidance)

        if do_correction:
            # Correction factor:
            V_est = V_est * (1 / (1-gammaEval)) / (1 / (1-gamma_guidance))

       # Evaluate performance of estimated value:
        evaluation_loss[i_gamma, i_rep] = np. abs(V_pi - V_est).mean()
        # test_loss[i_gamma, i_rep] = -V_CE_test.mean()
        # train_loss[i_gamma, i_rep] = -V_CE_train.mean()

plt.figure()
ci_factor = 1.96/np.sqrt(n_reps)  # 95% confidence interval factor
plt.errorbar(gamma_grid, evaluation_loss.mean(axis=1), yerr=evaluation_loss.std(axis=1) * ci_factor,
             fmt='b.', label='{} trajectories'.format(n))
plt.title(method + ' - Evaluation Loss \n' r'(Average absolute estimation error of $V^{\pi}(s)$)''\n (+- 95% confidence interval)')
plt.grid(True)
plt.xlabel('Guidance Discount Factor')
plt.ylabel('Planing Loss ')
plt.legend()

plt.show()
print('done')
