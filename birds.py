import numpy as np
from numpy.random import random, randint
from scipy.spatial import KDTree

STEPSIZE = 0.3

def circular_mean(dirs):
    return np.arctan2(sum(np.sin(dirs))/len(dirs),sum(np.cos(dirs))/len(dirs))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

class Birds(object):
    def __init__(self, numbirds, field_dims, res):
        self.numbirds = numbirds

        # Initialization
        self.positions = [
            np.array([
                randint(*(res * field_dims[0:2])),
                randint(*(res * field_dims[2:4]))
            ]) / res for _ in range(numbirds)
        ]
        print(self.positions)
        self.instincts = 2 * np.pi * random(self.numbirds) # Initial instinct
        self.dirs = self.instincts # Initial flight angle

    def update(self, r_0 = STEPSIZE):
        new_dirs = self.instincts # Constant flight direction

        # Update flight directions and instincts
        self.dirs = new_dirs

        # Update positions
        self.positions += np.array(
            [[r_0 * np.cos(theta), r_0 * np.sin(theta)] for theta in self.dirs]
        )
