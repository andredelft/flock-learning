import os
from os import path
import sys
from glob import glob
import numpy as np
import json
import regex

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plot as p
from utils import get_rt, shiftedColorMap

# When working from another directory
if __name__ == "__main__":
    HERE = path.split(path.abspath(__file__))[0]
else:
    HERE = path.abspath('.')

DATA_DIR = path.join(HERE, 'data')
IMG_DIR = path.join(HERE, 'thesis-figures')

# mpl.rcParams['font.serif'] = ['Palatino']
mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['mathtext.it'] = 'Brill'
# mpl.rcParams['mathtext.bf'] = 'Brill:bold'
# mpl.rcParams['mathtext.rm'] = 'Brill'
mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['figure.figsize'] = [5, 4]
# mpl.rcParams["savefig.format"] = '.pdf'
# mpl.rcParams['savefig.directory'] = HERE

vmin, vmax = (0.2, 1)

for name in ['Blues', 'Reds', 'Greens']:
    shiftedColorMap(
        plt.get_cmap(name), vmin, (vmax + vmin)/2, vmax, f'Shifted{name}'
    )

CMAP = {
    'alpha': plt.get_cmap('ShiftedBlues'),
    'gamma': plt.get_cmap('ShiftedReds'),
    'epsilon': plt.get_cmap('ShiftedGreens'),
    'references': plt.get_cmap('Greys')
}

def get_path(fname):
    date = regex.search('^\d*', fname).group()
    return path.join(DATA_DIR, date, fname)

DELTA_T = 500

def plot_delta(fig, fname, data_dir = '', delta_t = DELTA_T, **kwargs):
    if data_dir:
        fpath = path.join(data_dir, fname)
    else:
        fpath = get_path(fname)
    data = np.load(fpath)
    fig.plot(range(0, delta_t * len(data), delta_t), data, **kwargs)

class Figures:

    def vDelta():
        fig,a = plt.subplots(2,1)
        dir1 = path.join(DATA_DIR, '20200424', '2-avg_v')
        dir2 = path.join(DATA_DIR, '20200425')
        data = []
        data.append(np.load(path.join(dir1, 'avg_v.npy')))
        data.append(np.load(path.join(dir2, 'avg_v.npy')))

        for i,d in enumerate(data):
            a[i].scatter(*d, marker = '.')
            a[i].set_ylim(0,1)
            a[i].set_ylabel(r'$|\mathbf{v}|$')
        a[1].set_xlabel(r'$\Delta$')
        a[0].set_xlim(0,1)
        a[1].set_xlim(0,0.5)
        plt.tight_layout()
        plt.savefig(path.join(IMG_DIR, 'v_delta.pdf'))

    def TtLP():
        fig,a = plt.subplots(2,2)

        ref_cvalue = 0.4
        ref_fnames = []
        par_fnames = {
            'alpha': [],
            'gamma': [],
            'epsilon': []
        }

        lp_data_dir = path.join(DATA_DIR, '20200519')
        fnames = sorted(fname for fname in os.listdir(lp_data_dir) if fname.endswith('-Delta.npy'))
        with open(path.join(lp_data_dir, 'parameters.json')) as f:
            params = json.load(f)

        for fname in fnames:
            pars = params[get_rt(fname)]
            comment = pars.pop('comment', '')
            if comment == 'reference':
                ref_fnames.append(fname)
            elif comment.startswith('vary_'):
                par_fnames[comment.split('_')[1]].append(fname)

        for i, fname in enumerate(ref_fnames):
            plot_delta(
                a[0][0], fname, data_dir = lp_data_dir, label = get_rt(fname),
                color = CMAP['references'](-0.7 * (i/len(ref_fnames)) + 0.9)
            )
            plot_delta(
                a[0][1], fname, data_dir = lp_data_dir, label = get_rt(fname),
                color = CMAP['references'](ref_cvalue)
            )
            plot_delta(
                a[1][0], fname, data_dir = lp_data_dir, label = get_rt(fname),
                color = CMAP['references'](ref_cvalue)
            )
            plot_delta(
                a[1][1], fname, data_dir = lp_data_dir, label = get_rt(fname),
                color = CMAP['references'](ref_cvalue)
            )

        axins = dict()
        for par, fignums in zip(['alpha', 'gamma', 'epsilon'], [(0,1), (1,0), (1,1)]):
            for fname in par_fnames[par]:
                record_tag = get_rt(fname)
                value = params[record_tag]['Q_params'][par]
                plot_delta(
                    a[fignums], fname, data_dir = lp_data_dir,
                    color = CMAP[par](value)
                )
            axins[par] = inset_axes(a[fignums], width='20%', height='3%', loc='upper right')
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(cmap = CMAP[par]), cax=axins[par],
                orientation='horizontal', ticks=[0, 1]
            )
            cbar.ax.set_ylabel(fr'$\{par}$  ', rotation = 0, y = -1)

        a[0][0].set_ylim(0.415, 0.505)
        a[0][1].set_ylim(0.415, 0.505)
        a[1][0].set_ylim(0.415, 0.505)
        a[1][1].set_ylim(0.415, 0.505)

        a[0][0].set_xticklabels([])
        a[0][1].set_xticklabels([])
        a[0][1].set_yticklabels([])
        a[1][1].set_yticklabels([])
        a[0][0].set_ylabel(r'$\Delta$')
        a[1][0].set_ylabel(r'$\Delta$')
        a[1][0].ticklabel_format(style = 'sci', axis = 'x', scilimits = (3,3))
        a[1][0].set_xlabel(r'Timestep')
        a[1][1].ticklabel_format(style = 'sci', axis = 'x', scilimits = (3,3))
        a[1][1].set_xlabel('Timestep')

        plt.tight_layout()
        plt.savefig(path.join(IMG_DIR, 'tweaking_the_learning_params.pdf'))


if __name__ == "__main__":
    if sys.argv[-1] == 'update_all' or len(sys.argv) == 1:
        for func in Figures.__dict__.keys():
            if not func.startswith('__'):
                getattr(Figures, func)()
    else:
        getattr(Figures, sys.argv[-1])()
