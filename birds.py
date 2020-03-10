import numpy as np
from random import randint, choice, choices
from scipy.spatial import KDTree

from q_learning import Qfunction

D = 100 # Observation radius
N = 2   # Max neigbours observed
R = 1   # Reward signal

alpha   = 0.9  # Learning rate
gamma   = 0.9  # Disount factor
epsilon = 0.2  # Epsilon-greedy parameter

A = ['N', 'E', 'S', 'W', 'I'] # Action space

STEP = {
    'N': np.array([ 0, 1]),
    'E': np.array([ 1, 0]),
    'S': np.array([ 0,-1]),
    'W': np.array([-1, 0])
}

ANG = {
    'N': np.pi/2,
    'E': 0,
    'S': -1 * np.pi/2,
    'W': np.pi
}

def discrete_Vicsek(observation):
    v = np.array([0,0])
    for dir,n in observation.items():
        for _ in range(n):
            v += STEP[dir]
    if list(v) != [0,0]:
        angle = np.arctan2(v[1],v[0])
        if (angle < -3 * np.pi / 4) or (angle >= 3 * np.pi / 4):
            return 'W'
        elif (angle < 3 * np.pi / 4) and (angle >= np.pi / 4):
            return 'N'
        elif (angle < np.pi / 4) and (angle >= -1 * np.pi / 4):
            return 'E'
        elif (angle < -1 * np.pi/4) and (angle >= -3 * np.pi / 4):
            return 'S'
    else:
        return 0

def ternary(numbers):
    numbers = list(numbers)
    return sum(numbers[-1 * (i + 1)] * 3 ** i for i in range(len(numbers)))

class Birds(object):
    def __init__(self, numbirds, field_dims, observe_direction = True,
                 leader_frac = 0.25, reward_signal = R, learning_alg = 'Ried',
                 alpha = alpha, gamma = gamma, epsilon = epsilon):

        self.numbirds = numbirds
        self.leaders = int(self.numbirds * leader_frac)
        self.observe_direction = observe_direction
        self.reward_signal = reward_signal
        self.epsilon = epsilon
        print(' '.join(['Simulation started with birds that observe the',
              'flight direction' if observe_direction else 'relative position',
              'of neighbours']))

        # Initialization of birds in the field
        self.positions = np.array([
            np.array([
                randint(*field_dims[0:2]), randint(*field_dims[2:4])
            ]) for _ in range(self.numbirds)
        ])
        self.dirs = choices(['N','E','S','W'], k = self.numbirds)
        self.instincts = self.leaders * ['E'] + choices(['N','S','W'], k = self.numbirds - self.leaders)
        self.policies = np.zeros([self.numbirds, (N + 1)**4, len(A)]) + 1/len(A)
        self.observations = [self.observe(i) for i in range(self.numbirds)]
        self.learning_alg = learning_alg
        if self.learning_alg == 'Q':
            self.Qs = [Qfunction(alpha, gamma, A = A, S = range((N + 1)**4)) for _ in range(self.numbirds)]


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

    def perform_step(self, step = STEP):
        for i in range(self.numbirds):
            if self.actions[i] == 'V':
                if not self.observe_direction:
                    raise Exception('Vicsek action is not allowed when observe_direction = False')
                else:
                    Vic_dir = discrete_Vicsek(self.observations[i])
                    if Vic_dir == 0:
                        # Maintain current direction
                        self.dirs[i] = self.dirs[i]
                    else:
                        self.dirs[i] = Vic_dir
            elif self.actions[i] == 'I':
                self.dirs[i] = self.instincts[i]
            elif self.actions[i] in ['N','E','S','W']:
                self.dirs[i] = self.actions[i]
            elif self.actions[i] == 'R':
                self.dirs[i] = choice(['N','E','S','W'])
            else:
                raise ValueError(f'Action {self.actions[i]} does not exist')
            self.positions[i] += step[self.dirs[i]]

    def reward(self,i):
        return self.reward_signal if self.dirs[i] == 'E' else 0

    def Ried_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.prev_obs[i].values())
            reward = self.reward(i)
            if reward:
                self.policies[i,s,A.index(self.actions[i])] += reward/100
                self.policies[i,s] = self.policies[i,s]/sum(self.policies[i,s])

    def Q_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.observations[i].values())
            s_prime = ternary(self.prev_obs[i].values())
            self.Qs[i].update(s, self.actions[i], s_prime, self.reward(i))
            argmax_Q = np.argmax(self.Qs[i].value[s])
            self.policies[i,s] = self.epsilon/len(A) + np.array([(1 - self.epsilon if j == argmax_Q else 0) for j in range(len(A))])

    def update(self):
        # print(self.policies[0])
        self.actions = [choices(
            A, weights = self.policies[i, ternary(self.observations[i].values())],
        )[0] for i in range(self.numbirds)]
        self.perform_step()
        self.prev_obs = self.observations
        self.observations = [self.observe(i) for i in range(self.numbirds)]
        if self.learning_alg == 'Ried':
            self.Ried_learning()
        elif self.learning_alg == 'Q':
            self.Q_learning()
