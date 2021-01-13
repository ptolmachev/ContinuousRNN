'''
A script containing a simple RNN description:
equations:
dh/dt = -h W sigma(h) + b

sigma(h) function described in 'state_function.py'
'''

from matplotlib import pyplot as plt
from copy import deepcopy
from collections import deque
from src.state_function import *


class RNN():
    def __init__(self, dt, lmbd, W, b):
        self.lmbd = lmbd
        self.dt = dt
        self.W = W
        self.b = b
        #number of neurons-nodes
        self.N = len(self.b)
        self.h = 10 * np.random.randn(self.N)

        self.t = 0
        self.h_history = deque()
        # self.h_history.append(deepcopy(self.h))

    #state function
    def state(self, h):
        return s(self.lmbd, h)

    def rhs(self):
        return -self.h + self.W @ self.state(self.h) + self.b

    def step(self):
        self.h = self.h + self.dt * self.rhs()
        self.t += self.dt
        return None

    def update_history(self):
        self.h_history.append(deepcopy(self.h))
        return None

    def run(self, T):
        N_steps = int(np.ceil(T/ self.dt))
        for i in (range(N_steps)):
            self.step()
            self.update_history()
        return None

    def plot_history(self):
        fig, ax = plt.subplots(self.N, 1, figsize=(15, 5))
        h_array = np.array(self.h_history)
        t_array = np.arange(h_array.shape[0]) * self.dt
        for i in range(self.N):
            ax[i].plot(t_array, h_array[:, i], linewidth=2, color='k')
            if (i == self.N//2):
                ax[i].set_ylabel(f'h', fontsize=24, rotation=0)
        ax[-1].set_xlabel('t', fontsize=24)
        plt.subplots_adjust(hspace=0)
        plt.suptitle(f"Trajectory of a neural network, N={self.N}, lmbd={self.lmbd}", fontsize=24)
        return fig, ax

if __name__ == '__main__':
    N = 5
    W = 0.1 * (np.random.randn(N, N))
    np.fill_diagonal(W, 0)
    b = 0.1 * np.random.randn(N)
    lmbd = 0.5
    dt = 0.05
    rnn = RNN(dt, lmbd, W, b)
    T = 100
    rnn.run(T)
    fig, ax = rnn.plot_history()
    plt.show(block=True)







