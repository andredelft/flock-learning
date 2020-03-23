from numpy.random import rand

""" Sample state and action space:
A = ['N','E','S','W']
S = [0,1,2,3,4] # Assume S is a list of its indices (e.g. S == list(range(len(S))))
"""

class Qfunction(object):
    
    def __init__(self, alpha, beta, A, S):
        if alpha < 0 or alpha >= 1:
            raise ValueError("0 <= alpha < 1 is required")
        self.alpha = alpha
        if beta < 0 or beta >= 1:
            raise ValueError("0 <= beta < 1 is required")
        self.beta = beta
        self.A = A
        self.S = S
        self.value = rand(len(self.S),len(self.A))

    def update(self, s, a, s_prime, r):
        a_ind = self.A.index(a)
        max_Q = max(self.value[s_prime, self.A.index(a_prime)] for a_prime in self.A)
        self.value[s, self.A.index(a)] = (1 - self.alpha) * self.value[s, a_ind] + self.alpha * (r + self.beta * max_Q)
