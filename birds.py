import numpy as np
from numpy.random import random
from scipy.spatial import KDTree

NUM_BIRDS = 50
NUM_NN = 6
INTIAL_FLOCKSIZE = 200
RADIUS = 10
STEPSIZE = 0.3

def circular_mean(dirs):
    return np.arctan2(sum(np.sin(dirs))/len(dirs),sum(np.cos(dirs))/len(dirs))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

class Birds(object):
    def __init__(self,numbirds=NUM_BIRDS):
        self.numbirds = numbirds

        # Bird parameters
        self.trusts = random(self.numbirds)
        self.compliances = sigmoid(50 * (self.trusts - 0.5))

        # Initialization
        self.positions = INTIAL_FLOCKSIZE * (random((self.numbirds, 2))-0.5)
        self.instincts = 2 * np.pi * random(self.numbirds) # Initial instinct
        self.dirs = self.instincts # Initial flight angle

    def vicsek_instinct(self, N=NUM_NN, R=RADIUS, r_0=STEPSIZE):
        tree = KDTree(self.positions)
        new_dirs = []
        new_instincts = []
        for i,bird in enumerate(self.positions):
            neighbours = sorted(
                tree.query_ball_point(bird, R),
                key = lambda j: np.linalg.norm(bird - self.positions[j])
            )[:N+1] # N nearest neigbours and itself

            # Circular mean of nearest neigbours (itself included)
            theta_avg = circular_mean([self.dirs[n] for n in neighbours])

            if random() < self.trusts[i]:
                new_dirs.append(theta_avg)
            else:
                new_dirs.append(self.instincts[i])

            new_instincts.append(
                np.arctan2(
                    (1 - self.compliances[i]) * np.sin(self.instincts[i])
                    + self.compliances[i] * np.sin(theta_avg),
                    (1 - self.compliances[i]) * np.cos(self.instincts[i])
                    + self.compliances[i] * np.cos(theta_avg),
                )
            )

        # Update flight directions and instincts
        self.dirs = np.array(new_dirs)
        self.instincts = np.array(new_instincts)

        # Update positions
        self.positions += np.array(
            [[r_0 * np.cos(theta), r_0 * np.sin(theta)] for theta in self.dirs]
        )
