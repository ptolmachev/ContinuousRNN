'''
description of sigma(h) state function'
'''

import numpy as np

def s(lmbd, h):
    return (2 / np.pi) * np.arctan(lmbd * h)

def der_s(lmbd, h):
    return (2 / np.pi) * lmbd * (1 / (1 + (lmbd * h) ** 2))

