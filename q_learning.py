from numpy.random import rand


class Qfunction(object):

    def __init__(self, alpha, gamma, S, A):
        """
        Initiate the Q-table of the agent, based on the following (required)
        arguments:

        alpha   : The learning rate, providing a weight for the old Q-values
                  relative to the new value.
        gamma   : The discount factor, providing a weight for expected future
                  reward signals. Specifically, the expected reward signal of n
                  timesteps ahead will have a weight of gamma^n.
        S       : The state space (or observation space) of the agent. It is
                  assumed that this is a range of indices, listing all possible
                  states.
        A       : The action space of the agent, which can be a list of any
                  desired type of objects. For example, in our models a list of
                  characters is used (e.g. A = ['N', 'E', 'S', 'W']).
                  NB: while this is not true in all possible applications, we
                  here assume that each action can be chosen in every available
                  state. This is sufficient for the purposes of our model.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("0 <= alpha <= 1 is required")
        self.alpha = alpha
        if gamma < 0 or gamma > 1:
            raise ValueError("0 <= gamma <= 1 is required")
        self.gamma = gamma
        self.A = A
        self.S = S
        self.table = rand(len(self.S),len(self.A))

    def update(self, s, a, s_prime, r):
        """
        Updates the Q-table, according to the following values:

        s       : The initial state that the agent has observed (element of S).
        a       : The action that the agent has chosen (element of A).
        s_prime : The new state the agent is in after performing its action
                  (element of S).
        r       : The reward the agent has received after this step (a number).
        """
        a_ind = self.A.index(a)
        max_Q = max(self.table[s_prime])    # Calculate expected future reward
                                            # given the next state s_prime
        self.table[s, a_ind] = (
            (1 - self.alpha) * self.table[s, a_ind]     # Previous value
            +     self.alpha * (r + self.gamma * max_Q) # Update with reward and
        )                                               # expected future reward
