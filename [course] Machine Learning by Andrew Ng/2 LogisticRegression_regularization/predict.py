import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    p = np.zeros(m)

    p=sigmoid(X.dot(theta))
    p[p>=0.5]=1
    p[p<0.5]=0

    return p
