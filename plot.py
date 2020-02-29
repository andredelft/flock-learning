from matplotlib import pyplot as plt
import numpy as np
from os import path
from glob import glob
import json
import regex

from birds import ternary

def avg(data, cap = 20):
    return [sum(data[i:i + cap])/cap for i in range(len(data) - cap)]

def plot_mag(fname, label = '', cap = 50):
    data = np.load(fname)
    plt.plot(avg([np.linalg.norm(v) for v in data], cap=cap),label=label)

def plot_vx(fname, label = '', cap = 50):
    data = np.load(fname)
    plt.plot(avg([v[0] for v in data], cap=cap),label=label)

def plot_arg(fname, label = '', cap = 100):
    data = np.load(fname)
    plt.plot(avg([np.arctan2(v[1],v[0]) for v in data], cap=cap),label=label)

def plot_mag_arg(fname):
    data = np.load(fname)
    fig,a = plt.subplots(2,1)
    a[0].set_title(path.split(fname)[1])
    a[0].plot(avg([np.linalg.norm(v) for v in data], cap=50))
    a[0].set_ylabel('$|\mathbf{v}|$')
    a[1].set_xlabel('Timestep')
    a[1].set_ylabel('Arg(v)')
    a[1].plot(avg([np.arctan2(v[1],v[0]) for v in data], cap=100))

maj_N = [
    ternary([1,0,0,0]),
    *range(ternary([2,0,0,0]), ternary([2,0,1,1]) + 1),
    *range(ternary([2,1,0,0]), ternary([2,1,1,1]) + 1)
]

maj_E = [
    ternary([0,1,0,0]),
    *range(ternary([0,2,0,0]), ternary([0,2,1,1]) + 1),
    *range(ternary([1,2,0,0]), ternary([1,2,1,1]) + 1)
]

maj_S = [
    ternary([0,0,1,0]),
    ternary([0,0,2,0]), ternary([0,0,2,1]),
    ternary([0,1,2,0]), ternary([0,1,2,1]),
    ternary([1,0,2,0]), ternary([1,0,2,1]),
    ternary([1,1,2,0]), ternary([1,1,2,1])
]

maj_W = [
    ternary([0,0,0,1]),
    ternary([0,0,0,2]), ternary([0,0,1,2]),
    ternary([0,1,0,2]), ternary([0,1,1,2]),
    ternary([1,0,0,2]), ternary([1,0,1,2]),
    ternary([1,1,0,2]), ternary([1,1,1,2])
]

def plot_hist(fname):
    pass

def plot_all(data_dir = 'data', quantity = 'mag', cap = 100):
    fnames = glob(f'{data_dir}/*.npy')
    with open(path.join(data_dir,'parameters.json')) as f:
        pars = json.load(f)
    for fname in fnames:
        label = regex.search(r'^[0-9]+-[0-9]+',path.split(fname)[1]).group()
        if quantity == 'mag':
            plot_mag(
                fname, cap = cap,
                label = 'Direction' if pars[label]['observe_direction'] else 'Position'
            )
    if quantity == 'mag':
        plt.title(f'Magnitude of average velocity vector (Capsize = {cap})')
        plt.ylabel('v')
    plt.xlabel('Timestep')
    plt.legend()
