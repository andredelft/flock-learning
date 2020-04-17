import os
import sys
sys.path.append(os.path.join(sys.path[0], '..', '..'))

import plot as p
from matplotlib import pyplot as plt

if __name__ == "__main__":
    cmap = plt.get_cmap('Blues')

    p.plot_Delta('20200416-190443-Delta.npy', label = 'R = 20', color = cmap(0.99))
    p.plot_Delta('../20200415/20200415-144158-Delta.npy', label = 'R = 10', color = cmap(0.8))
    p.plot_Delta('../20200415/20200415-154211-Delta.npy', color = cmap(0.8))
    p.plot_Delta('20200416-092043-Delta.npy', color = cmap(0.4))
    p.plot_Delta('20200416-125734-Delta.npy', color = cmap(0.6), label = 'R = 5')
    p.plot_Delta('../20200415/20200415-122210-Delta.npy', color = cmap(0.4), label = 'R = 1')
    p.plot_Delta('20200416-114526-Delta.npy', color = cmap(0.4))

    plt.legend()
    plt.title('Gradient reward space for different max. reward signals')
    plt.xlabel('Delta')
    plt.ylabel('Timestep')
    plt.show()
