import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from os import path
import json

DATA_DIR = ''

def get_record_tag(fpath):
    fname = path.split(fpath)[1]
    return '-'.join(fname.split('-')[:2])

def calc_avg_v(vs):
    mag_v = [v[0]**2 + v[1]**2 for v in vs]
    return sum(mag_v[500:])/len(mag_v[500:])

def gen_fig(data_dir = DATA_DIR, save_file = True, expose_remote = False):
    avg_v = []
    Delta = []
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)
    for fname in sorted(glob(path.join(data_dir, '*-v.npy'))):
        V = np.load(fname)
        avg_v.append(calc_avg_v(V))
        record_tag = get_record_tag(fname)
        Delta.append(params[record_tag]['Delta'])

    np.save(path.join(data_dir, 'avg_v.npy'), [Delta, avg_v])
    plt.figure()
    plt.plot(Delta, avg_v)
    plt.xlabel('$\\Delta$')
    plt.ylabel('Avg(v)')
    if save_file:
        if expose_remote:
            data_dir = path.join(path.expanduser('~'), 'public_html')
        plt.savefig(path.join(data_dir, 'avg_v.png'), dpi = 300)

if __name__ == "__main__":
    gen_fig()
