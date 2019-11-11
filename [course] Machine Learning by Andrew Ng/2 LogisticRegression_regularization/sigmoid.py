import numpy as np


def sigmoid(z):
    g = np.zeros(z.size)

    g=1/(1+np.exp(-z))

    return g
