import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))

import plot as p
from matplotlib import pyplot as plt
import numpy as np
import json

cmap = {
    'alpha': plt.get_cmap('Blues'),
    'gamma': plt.get_cmap('Reds'),
    'epsilon': plt.get_cmap('Greens')
}   

if __name__ == "__main__":

    plt.figure()

    p.plot_Delta('20200417/20200417-131821-Delta.npy', color = 'Darkgrey', label = 'Reference')
    # p.plot_Delta('20200418/20200418-114214-Delta.npy', color = 'Darkgrey')
    p.plot_Delta('20200422-152117-Delta.npy', color = 'Black')
    # p.plot_Delta('20200422-160846-Delta.npy', color = 'Black')

    fnames = sorted(fname for fname in os.listdir() if fname.endswith('-Delta.npy'))
    
    with open('parameters.json') as f:
        params = json.load(f)

    for fname in fnames:
        record_tag = '-'.join(fname.split('-')[:2])
        if 'comment' in params[record_tag].keys():
            par_name = params[record_tag]['comment'].split('_')[1]
            value = params[record_tag]['Q_params'][par_name]
            p.plot_Delta(fname, color = cmap[par_name](value * 1.1), label = f'$\\{par_name} = {value}$')

    p.plot_Delta('20200422-160846-Delta.npy', color = 'Black')
    plt.legend()
    # plt.xlim(0,20000)
    # plt.ylim(0.44, 0.5)
    plt.xlabel('Timestep')
    plt.ylabel('$\Delta$')
    plt.savefig('../../../public_html/Delta.png')
