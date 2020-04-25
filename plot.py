from matplotlib import pyplot as plt
# from matplotlib import style
import numpy as np
from os import path
from glob import glob
import json
import pickle
import regex

from birds import discrete_Vicsek, CARD_DIRS

re_tag = regex.compile(r'^[0-9]+-[0-9]+|^desired')

def _find_tag(fpath):
    fname = path.split(fpath)[1]
    return re_tag.search(fname).group()

def _parse_fpath(fpath, data_dir):
    record_tag = _find_tag(fpath)
    if fpath == record_tag:
        data_dir = 'data'
        fname = ''
    else:
        data_dir, fname = path.split(fpath)
    return data_dir, fname, record_tag

def avg(data, cap = 20):
    return [sum(data[i:i + cap])/cap for i in range(len(data) - cap)]

def plot_mag(fname, cap = 50, **kwargs):
    data = np.load(fname)
    plt.plot(avg([np.linalg.norm(v) for v in data], cap = cap), **kwargs)

def plot_vx(fname, label = '', cap = 50):
    data = np.load(fname)
    plt.plot(avg([v[0] for v in data], cap = cap), label = label)

def plot_arg(fname, label = '', cap = 100):
    data = np.load(fname)
    plt.plot(avg([np.arctan2(v[1],v[0]) for v in data], cap = cap), label = label)

def plot_mag_arg(fname):
    data = np.load(fname)
    fig,a = plt.subplots(2,1)
    a[0].set_title(path.split(fname)[1])
    a[0].plot(avg([np.linalg.norm(v) for v in data], cap = 50))
    a[0].set_ylabel('$|\mathbf{v}|$')
    a[1].set_xlabel('Timestep')
    a[1].set_ylabel('Arg(v)')
    a[1].plot(avg([np.arctan2(v[1],v[0]) for v in data], cap = 100))

def plot_Delta(fname, **kwargs):
    data = np.load(fname)
    plt.plot(range(0, 500 * len(data), 500), data, **kwargs)

def plot_all(data_dir = 'data', quantity = 'v', cap = 50, **kwargs):
    with open(path.join(data_dir,'parameters.json')) as f:
        pars = json.load(f)
    for fname in sorted(glob(f'{data_dir}/*-{quantity}.npy')):
        record_tag = _find_tag(fname)
        if quantity == 'v':
            plot_mag(
                fname, cap = cap,
                label = record_tag,
                **kwargs
            )
        elif quantity == 'Delta':
            plot_Delta(
                fname, label = record_tag,
                **kwargs
            )
    if quantity == 'v':
        plt.title(f'Magnitude of average velocity vector (Capsize = {cap})')
        plt.ylabel('v')
    elif quantity == 'Delta':
        plt.ylabel('$\Delta$')
    plt.xlabel('Timestep')
    plt.legend()
