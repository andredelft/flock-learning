import numpy as np
from random import randint, choice, choices
from scipy.spatial import KDTree
from bisect import bisect_left

from q_learning import Qfunction

D = 100 # Observation radius
N = 2   # Max neigbours observed
R = 1   # Reward signal

alpha   = 0.9  # Learning rate
gamma   = 0.9  # Disount factor
epsilon = 0.2  # Epsilon-greedy parameter

A = ['V', 'I'] # Action space

NO_DIRS = 2 ** 2 # exponent should be >= 2
DIRS = np.linspace(-np.pi, np.pi, NO_DIRS + 1)[:NO_DIRS]
STEPS = [np.array([np.cos(theta), np.sin(theta)]) for theta in DIRS]
TRESHOLD = 0.9 * np.linalg.norm(STEPS[0] + STEPS[(NO_DIRS//2) + 1])

CARD_DIRS = {card_dir: i for card_dir, i in zip('WSEN',range(0, NO_DIRS, NO_DIRS//4))}

def discrete_Vicsek(observation):
    """ Returns the index of DIRS that is closest to the average direction """
    v = np.array([0,0])
    for dir,n in observation.items():
        v += n * STEPS[dir]
    if np.linalg.norm(v) > TRESHOLD:
        theta = np.arctan2(v[1],v[0])
        i = bisect_left(DIRS, theta)
        if i == NO_DIRS - 1:
            if theta - DIRS[i] < 2 * np.pi - theta:
                return i
            else:
                return 0
        elif theta - DIRS[i] < theta - DIRS[i + 1]:
            return i
        else:
            return i + 1
    else:
        # Sum of dirs is zero
        return NO_DIRS

def ternary(numbers):
    numbers = list(numbers)
    return sum(numbers[-1 * (i + 1)] * 3 ** i for i in range(len(numbers)))

class Birds(object):

    def __init__(self, numbirds, field_dims, action_space = A, observe_direction = True,
                 leader_frac = 0.25, reward_signal = R, learning_alg = 'Ried',
                 alpha = alpha, gamma = gamma, epsilon = epsilon, Q_file = ''):

        self.numbirds = numbirds
        self.action_space = action_space
        self.leader_frac = leader_frac
        self.leaders = int(self.numbirds * self.leader_frac)
        self.observe_direction = observe_direction
        self.reward_signal = reward_signal
        self.learning_alg = learning_alg

        print(' '.join([
            'Simulation started with birds that observe the',
            'flight direction' if observe_direction else 'relative position',
            'of neighbours'
        ]))

        # Initialization of birds in the field
        self.positions = np.array([
            np.array([
                randint(*field_dims[0:2]), randint(*field_dims[2:4])
            ]) for _ in range(self.numbirds)
        ])
        self.dirs = choices(['N','E','S','W'], k = self.numbirds)
        self.instincts = self.leaders * ['E'] + choices(['N','S','W'], k = self.numbirds - self.leaders)
        self.policies = np.zeros([self.numbirds, (N + 1)**4, len(self.action_space)])
        if self.learning_alg == 'pol_from_Q':
            if not Q_file:
                raise Exception('No file with Q-values supplied')
            else:
                Qvalues = np.load(Q_file)
                for i in range(self.numbirds):
                    for s in range(self.policies.shape[1]):
                        self.policies[i,s,np.argmax(Qvalues[i,s])] = 1
        else:
            self.policies +=  1/len(self.action_space) # Fill all policy matrices
        self.observations = [self.observe(i) for i in range(self.numbirds)]

        if self.learning_alg == 'Q':
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.Qs = [Qfunction(
                alpha, gamma, A = self.action_space, S = range((N + 1)**4)) for _ in range(self.numbirds
            )]

    def request_params(self):
        params = {
            'no_birds': self.numbirds,
            'action_space': self.action_space,
            'observe_direction': self.observe_direction,
            'leader_frac': self.leader_frac,
            'reward_signal': self.reward_signal,
            'learning_alg': self.learning_alg
        }
        if self.learning_alg == 'Q':
            params['Q_params'] = {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        return params

    def calc_v(self):
        return sum(STEP[dir] for dir in self.dirs)/self.numbirds

    def observe(self, bird_index, radius = D):
        tree = KDTree(self.positions)
        neighbours_inds = tree.query_ball_point(self.positions[bird_index], radius)
        neighbours_inds.remove(bird_index)
        neighbours = {i: 0 for i in range(NO_DIRS)}

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
                if not self.observe_direction:
                    raise Exception('Vicsek action is not allowed when observe_direction = False')
                else:
                    i_dir = discrete_Vicsek(self.observations[i])
                    if i_dir == NO_DIRS:
                        # Maintain current direction
                        pass
                    else:
                        self.dirs[i] = i_dir
            elif self.actions[i] == 'I':
                self.dirs[i] = CARD_DIRS[self.instincts[i]]
            elif self.actions[i] in ['N','E','S','W']:
                self.dirs[i] = CARD_DIRS[self.actions[i]]
            elif self.actions[i] == 'R':
                self.dirs[i] = choice(DIRS)
            else:
                raise ValueError(f'Action {self.actions[i]} does not exist')
            self.positions[i] += STEP[self.dirs[i]]

    def reward(self,i):
        return self.reward_signal if self.dirs[i] == 'E' else 0

    def Ried_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.prev_obs[i].values())
            reward = self.reward(i)
            if reward:
                self.policies[i,s,self.action_space.index(self.actions[i])] += reward/100
                self.policies[i,s] = self.policies[i,s]/sum(self.policies[i,s])

    def Q_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.observations[i].values())
            s_prime = ternary(self.prev_obs[i].values())
            self.Qs[i].update(s, self.actions[i], s_prime, self.reward(i))
            argmax_Q = np.argmax(self.Qs[i].value[s])
            self.policies[i,s] = self.epsilon/len(self.action_space) + np.array([
                (1 - self.epsilon if j == argmax_Q else 0) for j in range(len(self.action_space))
            ])

    def update(self):
        # print(self.policies[80])
        self.actions = [choices(
            self.action_space, weights = self.policies[i, ternary(self.observations[i].values())]
        )[0] for i in range(self.numbirds)]
        self.perform_step()
        self.prev_obs = self.observations
        self.observations = [self.observe(i) for i in range(self.numbirds)]
        if self.learning_alg == 'Ried':
            self.Ried_learning()
        elif self.learning_alg == 'Q':
            self.Q_learning()
