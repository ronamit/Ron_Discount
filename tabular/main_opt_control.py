"""
In  tabular MDP setting, evaluates the learning of optimal policy using different guidance discount factors
"""
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
from tabular.data_utils import GenerateMDP, SampleTrajectories
from tabular.planing_utils import GetUniformPolicy, GetPolicyDynamics, PolicyEvaluation, PolicyIteration
from tabular.learning_utils import ModelEstimation

###############  Model definition   ##################
S = 10  # number of states
A = 2  # number of actions
k = 5  #  Number of non-zero entries in each row  of transition-matrix
reward_std = 0.1
gammaEval = 0.99

n = 5  # number of trajectories to generate
depth = 10  # Length of trajectory

n_reps = 10  # number of experiment repetitions

##############  Optimal Control (as in Jiang '15)   ##################


gamma_grid = np.linspace(0.1, 0.99, num=20)
n_gammas = gamma_grid.shape[0]

train_loss = np.zeros((n_gammas, n_reps))
test_loss = np.zeros((n_gammas, n_reps))
planing_loss = np.zeros((n_gammas, n_reps))

for i_rep in range(n_reps):
    # Generate MDP:
    P, R = GenerateMDP(S, A, k)

    # Optimal policy for the MDP:
    pi_opt, V_opt, Q_opt = PolicyIteration(P, R, gammaEval)

    # Set a fixed exploration policy pi  -  uniform over actions - for gathering data
    pi = GetUniformPolicy(S, A)

    # Generate data:
    data = SampleTrajectories(P, R, pi, n, depth, p0=None, reward_std=reward_std)

    for i_gamma, gamma_guidance in enumerate(gamma_grid):
        # Estimate model:
        P_est, R_est = ModelEstimation(data, S, A)
        # CE policy w.r.t model-estimation and gamma_guidance:
        pi_CE,  _, _ = PolicyIteration(P_est, R_est, gamma_guidance)

        # Evaluate performance of CE policy:
        V_CE_train, _ = PolicyEvaluation(P_est, R_est, pi_CE, gammaEval)
        V_CE_test, _ = PolicyEvaluation(P, R, pi_CE, gammaEval)
        test_loss[i_gamma, i_rep] = -V_CE_test.mean()
        train_loss[i_gamma, i_rep] = -V_CE_train.mean()
        planing_loss[i_gamma, i_rep] = (np.abs(V_opt - V_CE_test)).mean() # (eq. 14 in Jiang, corrected to have abs)


ci_factor = 1.96/np.sqrt(n_reps)  # 95% confidence interval factor
plt.figure()
plt.errorbar(gamma_grid, train_loss.mean(axis=1), yerr=train_loss.std(axis=1) * ci_factor , fmt='g.', label='Train Loss')
plt.errorbar(gamma_grid, test_loss.mean(axis=1), yerr=test_loss.std(axis=1) * ci_factor, fmt='b.',  label='Test Loss')
plt.title('CE Policy Performance, #trajectories={}\n (+- 95% confidence interval)'.format(n))
plt.grid(True)
plt.xlabel('Guidance Discount Factor')
plt.ylabel('Minus Avg. Value')
plt.legend()

plt.figure()
ci_factor = 1.96/np.sqrt(n_reps)  # 95% confidence interval factor
plt.errorbar(gamma_grid, planing_loss.mean(axis=1), yerr=planing_loss.std(axis=1) * ci_factor,
             fmt='b.', label='{} trajectories'.format(n))
plt.title(r'Planing Loss (Average absolute estimation error of $V^*(s)$)'' \n (+- 95% confidence interval)')
plt.grid(True)
plt.xlabel('Guidance Discount Factor')
plt.ylabel('Planing Loss ')
plt.legend()

plt.show()
print('done')
