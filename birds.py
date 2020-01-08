import numpy as np
from numpy.random import random, randint
import random
from scipy.spatial import KDTree

A = ['N','E','S','W'] #Flight directions

def circular_mean(dirs):
    return np.arctan2(sum(np.sin(dirs))/len(dirs),sum(np.cos(dirs))/len(dirs))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

class Birds(object):
    def __init__(self, numbirds, field_dims, res):
        self.numbirds = numbirds

        # Initialization
        self.positions = np.array([
            np.array([
                randint(*(res * field_dims[0:2])),
                randint(*(res * field_dims[2:4]))
            ]) / res for _ in range(self.numbirds)
        ])

    def update(self):
        self.dirs = [random.choice(A) for _ in range(self.numbirds)]
