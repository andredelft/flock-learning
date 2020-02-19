import numpy as np
from random import randint, choice, choices
from scipy.spatial import KDTree

D = 100 # Observation radius
N = 2   # Max neigbours observed
R = 1   # Reward signal

A = ['I','V'] # Action space

step = {
    'N': np.array([ 0, 1]),
    'E': np.array([ 1, 0]),
    'S': np.array([ 0,-1]),
    'W': np.array([-1, 0])
}

def circular_mean(dirs):
    return np.arctan2(sum(np.sin(dirs))/len(dirs),sum(np.cos(dirs))/len(dirs))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def ternary(numbers):
    numbers = list(numbers)
    return sum(numbers[-1 * (i + 1)] * 3 ** i for i in range(len(numbers)))

class Birds(object):
    def __init__(self, numbirds, field_dims, observe_direction = False):
        self.numbirds = numbirds
        self.observe_direction = observe_direction
        print(' '.join(['Simulation started with birds that observe the',
              'flight direction' if observe_direction else 'relative position',
              'of neighbours']))

        # Initialization of birds in the field
        self.positions = np.array([
            np.array([
                randint(*field_dims[0:2]), randint(*field_dims[2:4])
            ]) for _ in range(self.numbirds)
        ])
        self.dirs = [choice(['N','E','S','W']) for _ in range(self.numbirds)]
        self.instincts = [choice(['N','E','S','W']) for _ in range(self.numbirds)]
        self.policies = np.zeros([self.numbirds, (N + 1)**4, len(A)]) + 100/len(A)

    def observe(self, bird_index, radius = D):
        tree = KDTree(self.positions)
        neighbours_inds = tree.query_ball_point(self.positions[bird_index], radius)
        neighbours_inds.remove(bird_index)
        neighbours = {
            'N': 0,
            'E': 0,
            'S': 0,
            'W': 0
        }

        # Two possible observations:
        # – Relative position of neighbours (direction_hist = False): Divides
        #   the observation space in four quadrants (N, E, S, W) and counts the
        #   number of birds in each quandrant.
        # - Flight direction of neighbours (direction_hist = True): Observes
        #   and counts the current flight direction of all birds in the
        #   observation space.
        if not self.observe_direction:
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
                    # Check if all cases are catched
                    raise ValueError(f'No value found for angle {angle}')
        else:
            for i in neighbours_inds:
                neighbours[self.dirs[i]] += 1

        # Maximum of 2
        for dir in neighbours.keys():
            if neighbours[dir] > 2:
                neighbours[dir] = 2
        return neighbours


    def perform_step(self):
        for i in range(self.numbirds):
            if self.actions[i] == 'V':
                self.dirs[i] = 'N'
            elif self.actions[i] == 'I':
                self.dirs[i] = self.instincts[i]
            else:
                raise ValueError(f'Action {self.actions[i]} does not exist')
            self.positions[i] += step[self.dirs[i]]

    def Ried_learning(self, reward_signal = R):
        for i in range(self.numbirds):
            if self.dirs[i] == 'E':
                j = ternary(self.observations[i].values())
                self.policies[i,j,A.index(self.actions[i])] += reward_signal
                self.policies[i,j] = 100 * self.policies[i,j]/sum(self.policies[i,j])

    def update(self):
        # print(self.policies[0])
        self.observations = [self.observe(i) for i in range(self.numbirds)]
        self.actions = [choices(
            A, weights = self.policies[i, ternary(self.observations[i].values())],
        )[0] for i in range(self.numbirds)]
        self.perform_step()
        self.Ried_learning()
