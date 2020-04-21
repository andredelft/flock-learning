from os import path
import sys
sys.path.append(path.join(sys.path[0], '..', '..'))

import plot as p
from matplotlib import pyplot as plt
import numpy as np
import json

cmap = plt.get_cmap('Reds')

def plot(fpath):
    data_dir, fname = path.split(fpath)
    trained_rt = '-'.join(fname.split('-')[:2])
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)
    record_tag = params[trained_rt]['comment']
    date = record_tag.split('-')[0]
    Delta = round(np.load(path.join('..', date, f'{record_tag}-Delta.npy'))[-1],3)
    print(Delta)
    # print(cmap(Delta))
    p.plot_mag(fpath, color = cmap((0.5 - Delta)/0.15), label = f'$\Delta = {Delta}$')

if __name__ == "__main__":
    plot('../20200420/20200420-123251-v.npy')
    plot('20200421-095555-v.npy')
    plot('../20200420/20200420-134027-v.npy')
    plot('../20200420/20200420-141506-v.npy')
    plt.legend()
    plt.show()
