import numpy as np
from numpy.random import random
from scipy.spatial import KDTree

NUM_BIRDS = 50
INITIAL_FLOCKSIZE = 200
STEPSIZE = 0.3

def circular_mean(dirs):
    return np.arctan2(sum(np.sin(dirs))/len(dirs),sum(np.cos(dirs))/len(dirs))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

class Birds(object):
    def __init__(self, numbirds = NUM_BIRDS, initial_flocksize = INITIAL_FLOCKSIZE):
        self.numbirds = numbirds

        # Initialization
        self.positions = initial_flocksize * (random((self.numbirds, 2))-0.5)
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
