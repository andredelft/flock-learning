from numpy.random import rand

# Sample state and action space
A = ['N','E','S','W']
S = [0,1,2,3,4]

class Qfunction(object):
    def __init__(self, alpha, beta, A = A, S = S):
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
        self.value[self.S.index(s), self.A.index(a)] = (1 - self.alpha) * self.value[self.S.index(s), self.A.index(a)] \
                                             + self.alpha * (r + self.beta * max(self.value[self.S.index(s_prime), self.A.index(a_prime)] for a_prime in A))
