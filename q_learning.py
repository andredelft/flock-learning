from numpy.random import rand

""" Sample state and action space:
A = ['N','E','S','W']
S = [0,1,2,3,4] # Assume S is a list of its indices (e.g. S == list(range(len(S))))
"""

class Qfunction(object):

    def __init__(self, alpha, gamma, A, S):
        if alpha < 0 or alpha >= 1:
            raise ValueError("0 <= alpha < 1 is required")
        self.alpha = alpha
        if gamma < 0 or gamma >= 1:
            raise ValueError("0 <= gamma < 1 is required")
        self.gamma = gamma
        self.A = A
        self.S = S
        self.value = rand(len(self.S),len(self.A))

    def update(self, s, a, s_prime, r):
        a_ind = self.A.index(a)
        max_Q = max(self.value[s_prime, self.A.index(a_prime)] for a_prime in self.A)
        self.value[s, self.A.index(a)] = (1 - self.alpha) * self.value[s, a_ind] + self.alpha * (r + self.gamma * max_Q)
