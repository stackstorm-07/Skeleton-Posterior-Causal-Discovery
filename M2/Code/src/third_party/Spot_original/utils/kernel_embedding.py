import numpy as np
import sklearn
PI = np.pi

class Kernel_Embedding(object):
    # kernel embedding from https://github.com/lopezpaz/causation_learning_theory
    def __init__(self, k=5, s=None, d=1): # in RCC k=100
        if not s: s = [0.15, 1.5, 15]
        self.k, self.s, self.d = k, s, d
        self.w = np.hstack(( # w in shape 2*15
            np.vstack([si * np.random.randn(k, d) for si in s]), # shape 15*1, first 5 rows~N(0, 0.15), then ~N(0, 1.5), ~N(0, 15)
            2 * PI * np.random.rand(k * len(s), 1) # shape 15*1, ~N(0, 2pi)
        )).T

    def get_empirical_embedding(self, a):
        # param: a (list) is the same as that in percentile(a, q): samples P_S to a distribution P
        if len(a) == 0: return [-1.] * self.k * len(self.s)      #np.ones((self.k * len(self.s))) * -1.
        arr = sklearn.preprocessing.scale(a)[:, None]            # arr = np.array(a)[:, None]
        return np.cos(np.dot(np.hstack((arr, np.ones((arr.shape[0], 1)))), self.w)).mean(axis=0)


def percentile_embedding(arr, k=11):
    if len(arr) == 0: return [-1.] * k
    offset = 100 / (k - 1)
    percentiles = [i * offset for i in range(k)]
    return np.percentile(arr, percentiles)    

def meanstdmaxmin(a):
    return [np.mean(a), np.std(a), np.max(a), np.min(a)] if len(a) else [-1., -1., -1., -1.]