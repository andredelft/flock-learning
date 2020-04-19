import os
import sys
sys.path.append(os.path.join(sys.path[0], '..', '..'))

import plot as p
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    cmap = plt.get_cmap('Blues')

    # Linear fit for tail of 20200416-190443
    start = 175_000
    data = np.load('20200416-190443-Delta.npy')[start//500:]
    [a, b], cov = np.polyfit(start + 500 * np.arange(len(data)), data, 1, cov = True)
    x = np.linspace(150_000, 410_000, 2)
    plt.plot(x, a * x + b, linestyle = '--', color = 'coral')
    print(f'y = {round(a, 10)} * x + {round(b, 2)}')
    print(cov)

    p.plot_Delta('../20200409/20200409-164937-Delta.npy', color = 'darkgray', label = 'Original')
    p.plot_Delta('../20200417/20200417-131821-Delta.npy', color = cmap(0.6))
    p.plot_Delta('20200416-190443-Delta.npy', label = 'R = 20', color = cmap(0.99))
    p.plot_Delta('../20200415/20200415-144158-Delta.npy', label = 'R = 10', color = cmap(0.8))
    p.plot_Delta('../20200415/20200415-154211-Delta.npy', color = cmap(0.8))
    p.plot_Delta('20200416-092043-Delta.npy', color = cmap(0.4))
    p.plot_Delta('20200416-125734-Delta.npy', color = cmap(0.6), label = 'R = 5')
    p.plot_Delta('../20200415/20200415-122210-Delta.npy', color = cmap(0.4), label = 'R = 1')
    p.plot_Delta('20200416-114526-Delta.npy', color = cmap(0.4))

    plt.legend()
    plt.title('Gradient reward space for different max. reward signals')
    plt.ylabel('$\Delta$')
    plt.xlabel('Timestep')
    plt.show()
