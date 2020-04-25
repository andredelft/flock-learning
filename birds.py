import numpy as np
from random import randint, choice, choices
from scipy.spatial import KDTree
from bisect import bisect_left
import time

from q_learning import Qfunction

D = 100 # Observation radius
N = 2   # Max neigbours observed
R = 1   # Reward signal

alpha   = 0.1  # Learning rate
gamma   = 0.9  # Disount factor
epsilon = 0.5  # Epsilon-greedy parameter

A = ['V', 'I'] # Action space

NO_DIRS = 2 ** 3 # exponent should be 2 or 3
DIRS = np.linspace(-np.pi, np.pi, NO_DIRS + 1)[:NO_DIRS]
DIRS_INDS = list(range(NO_DIRS))
STEPS = [np.array([np.cos(theta), np.sin(theta)]) for theta in DIRS]
TRESHOLD = 0.9 * np.linalg.norm(STEPS[0] + STEPS[(NO_DIRS//2) + 1])

S = range((N + 1) ** NO_DIRS) # Observation space

CARD_DIRS = {
    card_dir: i for card_dir, i in zip('WSEN',range(0, NO_DIRS, NO_DIRS//4))
}

def discrete_Vicsek(observation):
    """ Returns the index of DIRS that is closest to the average direction """
    v = np.array([0.,0.])
    for dir,n in observation.items():
        v += n * STEPS[dir]
    if np.linalg.norm(v) > TRESHOLD:
        theta = np.arctan2(v[1],v[0])
        i = bisect_left(DIRS, theta)
        if i == NO_DIRS:
            delta_1 = theta - DIRS[i - 1]
            delta_2 = 2 * np.pi - theta
            if delta_1 < delta_2:
                return i - 1
            elif delta_1 == delta_2:
                return choice([i - 1, i])
            else:
                return 0
        elif i == 0:
            # Since -pi < theta < pi, this only happens when theta = -pi
            return 0
        else:
            delta_1 = theta - DIRS[i - 1]
            delta_2 = DIRS[i] - theta
            if delta_1 < delta_2:
                return i - 1
            elif delta_1 == delta_2:
                # Important to avoid a biased direction
                return choice([i - 1, i])
            else:
                return i
    else:
        # Sum of dirs is below treshold (i.e., zero)
        return NO_DIRS

def ternary(numbers):
    numbers = list(numbers)
    return sum(numbers[-1 * (i + 1)] * 3 ** i for i in range(len(numbers)))

class Birds(object):

    def __init__(self, numbirds, field_dims, action_space = A, state_space = S,
                 leader_frac = 0.25, reward_signal = R, learning_alg = 'Ried',
                 alpha = alpha, gamma = gamma, epsilon = epsilon, Q_file = '',
                 Q_tables = None, gradient_reward = False, instincts = []):

        # Initialize the birds and put them in the field
        self.numbirds = numbirds
        self.leader_frac = leader_frac
        self.leaders = int(self.numbirds * self.leader_frac)
        self.positions = np.array([
            np.array([
                float(randint(*field_dims[0:2])),
                float(randint(*field_dims[2:4]))
            ]) for _ in range(self.numbirds)
        ])
        self.dirs = choices(DIRS_INDS, k = self.numbirds)
        if instincts:
            if len(instincts) != self.numbirds:
                raise ValueError(
                    'Given list of instincts does not equal number of birds'
                )
            else:
                self.instincts = instincts
        else:
            self.instincts = (
                self.leaders * ['E']
                + choices(['N','S','W'], k = self.numbirds - self.leaders)
            )

        # Initialize all the reinforcement learning objects objects
        self.action_space = action_space
        self.state_space = state_space
        self.reward_signal = reward_signal
        self.gradient = gradient_reward
        self.learning_alg = learning_alg
        self.policies = np.zeros(
            [self.numbirds, len(S), len(self.action_space)]
        )
        if self.learning_alg == 'pol_from_Q':
            if (not Q_file) and (type(Q_tables) != np.ndarray):
                raise Exception('No Q-values provided')
            elif Q_file:
                Q_tables = np.load(Q_file)
            else:
                for i in range(self.numbirds):
                    for s in range(self.policies.shape[1]):
                        self.policies[i,s,np.argmax(Q_tables[i,s])] = 1
            self.Q_tables = Q_tables
            self.Delta = self.calc_Delta()
        else:
            self.policies +=  1/len(self.action_space) # Fill all policy matrices
        self.observations = [dict() for _ in range(self.numbirds)]
        self._perform_observations()

        if self.learning_alg == 'Q':
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.Qs = [Qfunction(
                alpha, gamma, self.state_space, self.action_space
            ) for _ in range(self.numbirds)]

    def request_params(self):
        params = {
            'no_birds': self.numbirds,
            'action_space': self.action_space,
            'leader_frac': self.leader_frac,
            'reward_signal': self.reward_signal,
            'learning_alg': self.learning_alg,
            'gradient_reward': self.gradient,
            'no_dirs': NO_DIRS
        }
        if self.learning_alg == 'Q':
            params['Q_params'] = {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        if self.learning_alg == 'pol_from_Q':
            params['Delta'] = self.Delta
        return params

    def _perform_observations(self, radius = D):
        tree = KDTree(self.positions)
        for i in range(self.numbirds):
            # Might still be optimized, since each pair of neighbouring birds
            # is handled twice. But depends on KDTree's performance
            neighbours_inds = tree.query_ball_point(self.positions[i], radius)
            neighbours_inds.remove(i)
            neighbours = {n: 0 for n in DIRS_INDS}

            for n in neighbours_inds:
                neighbours[self.dirs[n]] += 1

            # Maximum of N
            for dir in neighbours.keys():
                if neighbours[dir] > N:
                    neighbours[dir] = N
            self.observations[i] = neighbours

    def _perform_step(self):
        for i in range(self.numbirds):
            if self.actions[i] == 'V':
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
                self.dirs[i] = choice(DIRS_INDS)
            else:
                raise ValueError(f'Action {self.actions[i]} does not exist')
            self.positions[i] += STEPS[self.dirs[i]]

    def reward(self, i):
        if self.gradient:
            # R * cos(theta)
            return self.reward_signal * STEPS[self.dirs[i]][0]
        else:
            return self.reward_signal if self.dirs[i] == CARD_DIRS['E'] else 0

    def Ried_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.prev_obs[i].values())
            reward = self.reward(i)
            if reward:
                a_ind = self.action_space.index(self.actions[i])
                self.policies[i,s,a_ind] += reward/100
                self.policies[i,s] = self.policies[i,s]/sum(self.policies[i,s])

    def Q_learning(self):
        for i in range(self.numbirds):
            s = ternary(self.observations[i].values())
            s_prime = ternary(self.prev_obs[i].values())

            # Update the Q-table
            self.Qs[i].update(s, self.actions[i], s_prime, self.reward(i))

            # Update the epsilon-greedy policy
            argmax_Q = np.argmax(self.Qs[i].table[s])
            self.policies[i,s] = (
                np.array([
                    (1 - self.epsilon if j == argmax_Q else 0)
                    for j in range(len(self.action_space))
                ]) # Chance of (1 - epsilon) to choose the action with the
                   # highest Q-value
                + self.epsilon/len(self.action_space) # Chance of epsilon to
            )                                         # choose a random action

    def calc_v(self):
        return sum(STEPS[dir] for dir in self.dirs)/self.numbirds

    def calc_Delta(self):
        if self.learning_alg not in ['Q', 'pol_from_Q'] or self.action_space != ['V','I']:
            return None

        Delta = 0
        for i in range(self.numbirds):
            desired_ind = 1 if self.instincts[i] == 'E' else 0
            for s in S:
                if self.learning_alg == 'Q':
                    if np.argmax(self.Qs[i].table[s]) != desired_ind:
                        Delta += 1
                elif self.learning_alg == 'pol_from_Q':
                    if np.argmax(self.Q_tables[i,s]) != desired_ind:
                        Delta += 1
        Delta /= self.numbirds * len(S)
        return Delta

    def update(self):
        """
        The main function that governs each update, to be called from outside.

        NB: Functions in between are provided in comments that track the time
        each step takes.
        """

        # print('New update')
        # t_start = time.perf_counter()
        self.actions = [choices(
            self.action_space,
            weights = self.policies[i, ternary(self.observations[i].values())]
        )[0] for i in range(self.numbirds)]
        # t_end = time.perf_counter()
        # print(f'Choosing actions: {round(t_end - t_start, 3)}')

        # t_start = time.perf_counter()
        self._perform_step()
        # t_end = time.perf_counter()
        # print(f'Perform step: {round(t_end - t_start, 3)}')

        # t_start = time.perf_counter()
        self.prev_obs = self.observations
        self._perform_observations()
        # t_end = time.perf_counter()
        # print(f'Observing: {round(t_end - t_start, 3)}')

        # t_start = time.perf_counter()
        if self.learning_alg == 'Ried':
            self.Ried_learning()
        elif self.learning_alg == 'Q':
            self.Q_learning()
        # t_end = time.perf_counter()
        # print(f'Learning: {round(t_end - t_start, 3)}')
