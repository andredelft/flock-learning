import numpy as np
from numpy.random import random, randint
import random
from scipy.spatial import KDTree

R = 100

A = ['N','E','S','W'] # Action space

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

    def observe(self, bird_index, radius=R):
        tree = KDTree(self.positions)
        neighbours_inds = tree.query_ball_point(self.positions[bird_index], radius)
        neighbours_inds.remove(bird_index)
        neighbours = {
            'N': 0,
            'E': 0,
            'S': 0,
            'W': 0
        }
        for i in neighbours_inds:
            delta_x = self.positions[bird_index,0] - self.positions[i,0]
            delta_y = self.positions[bird_index,1] - self.positions[i,1]
            angle = np.arctan2(delta_y,delta_x)
            if (angle < -3 * np.pi / 4) or (angle >= 3 * np.pi / 4):
                neighbours['E'] += 1
            elif (angle < 3 * np.pi / 4) and (angle >= np.pi / 4):
                neighbours['N'] += 1
            elif (angle < np.pi / 4) and (angle >= -1 * np.pi / 4):
                neighbours['W'] += 1
            elif (angle < -1 * np.pi/4) and (angle >= -3 * np.pi / 4):
                neighbours['S'] += 1
            else:
                raise ValueError(f'No value found for {angle}')
        return neighbours

    def update(self):
        print(self.observe(0))
        self.dirs = ['N' if i == 0 else random.choice(A) for i in range(self.numbirds)]
