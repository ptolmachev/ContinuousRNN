'''
A script which fins such W and b which produces oscillations
generates a random W and b, calculates fixed points, check the Jacobians around the fixed points and
saves the values if all the fixed points are unstable.
'''

import numpy as np
from src.RNN import RNN
from matplotlib import pyplot as plt
from src.stability_analysis import *
import warnings
warnings.filterwarnings("ignore")
import pickle
import os

N = 5
lmbd = 0.5
dt = 0.1
stable = 1
while stable > 0:
    W = 10 * np.random.randn(N, N)
    np.fill_diagonal(W, 0)
    b = np.random.randn(N)

    # stability analysis
    fps = calc_equilibria(lmbd, W, b)
    print(f'number of fixed points: {len(fps)}')
    stable = 0
    unstable = 0
    for fp in fps:
        jac = calculate_Jacobian(fp, lmbd, W, b)
        res = eig(jac)
        largest_real_part = np.max(np.real(res[0]))
        if largest_real_part < 0:
            stable += 1
        else:
            unstable += 1

#save the parameters to a picke file
param_dict = dict()
param_dict["lmbd"] = lmbd
param_dict["W"] = W
param_dict["b"] = b
param_dict["N"] = N
file_name = f"params_{N}_{lmbd}.pkl"
pickle.dump(param_dict, open(os.path.join("../", "data", file_name), "wb+"))

rnn = RNN(dt, lmbd, W, b)
T = 100
rnn.run(T)
fig, ax = rnn.plot_history()
plt.show(block=True)
