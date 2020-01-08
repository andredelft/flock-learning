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
    def __init__(self, numbirds, field_dims):
        self.numbirds = numbirds

        # Initialization
        self.positions = np.array([
            np.array([
                randint(*field_dims[0:2]), randint(*field_dims[2:4])
            ]) for _ in range(self.numbirds)
        ])

    def update(self):
        self.dirs = [random.choice(A) for _ in range(self.numbirds)]
