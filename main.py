from field import Field
from matplotlib import pyplot as plt
import numpy as np

def avg(data, cap = 20):
    return [sum(data[i:i + cap])/cap for i in range(len(data) - cap)]

def plot_mag(fname, label = ''):
    data = np.load(fname)
    plt.plot(avg([np.linalg.norm(v) for v in data], cap=50),label=label)

def plot_vx(fname, label = ''):
    data = np.load(fname)
    plt.plot(avg([v[0] for v in data], cap=50),label=label)

def plot_arg(fname, label = ''):
    data = np.load(fname)
    plt.plot(avg([np.arctan2(v[1],v[0]) for v in data], cap=100),label=label)

if __name__ == '__main__':
    Field(100, periodic = True, plot = False, record_data = True, observe_direction = True)
