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
mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['figure.figsize'] = [5, 4]
# mpl.rcParams["savefig.format"] = '.pdf'
mpl.rcParams['savefig.directory'] = IMG_DIR

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

def plot_delta(ax, fname, data_dir = '', delta_t = DELTA_T, **kwargs):
    if data_dir:
        fpath = path.join(data_dir, fname)
    else:
        fpath = get_path(fname)
    data = np.load(fpath)
    ax.plot(range(0, delta_t * len(data), delta_t), data, **kwargs)

class Figures:

    def v_delta():
        fig,a = plt.subplots(2,1)
        dir1 = path.join(DATA_DIR, '20200424', '2-avg_v')
        dir2 = path.join(DATA_DIR, '20200425')
        data = []
        data.append(np.load(path.join(dir1, 'avg_v.npy')))
        data.append(np.load(path.join(dir2, 'avg_v.npy')))

        for i,d in enumerate(data):
            a[i].scatter(*d, marker = '.')
            a[i].set_ylim(0,1)
            a[i].set_ylabel(r'$\langle v \rangle$')
        a[1].set_xlabel(r'$\Delta$')
        a[0].set_xlim(0,1)
        a[1].set_xlim(0,0.5)
        plt.tight_layout()
        plt.savefig(path.join(IMG_DIR, 'v_delta.pdf'))

    def optpol_leader_fractions():
        fig, ax = plt.subplots()
        data_dir = path.join(DATA_DIR, '20200406')
        with open(path.join(data_dir, 'parameters.json')) as f:
            params = json.load(f)
        # record_tags = [rt for rt in params.keys() if not rt.startswith('desired')]
        fpaths = glob(f'{data_dir}/*-v.npy')

        vmin, vmax = (0.18, 0.9)
        cmap_label = 'YlOrRd'
        shiftedColorMap(
            plt.get_cmap(cmap_label), vmin, (vmax + vmin)/2, vmax, f'Shifted{cmap_label}'
        )
        lf_cmap = plt.get_cmap(cmap_label)

        for fpath in sorted(fpaths):
            record_tag = get_rt(fpath)
            if 'comment' in params[record_tag].keys() and params[record_tag]['comment'].startswith('desired'):
                # data = np.load(fpath)
                p.plot_mag(fpath, c = lf_cmap(
                    (vmax - vmin) * (params[record_tag]['leader_frac'] - 0.05) / 0.25 + vmin
                ), max = 900)

        axins = inset_axes(ax, width = '100%', height = '100%', bbox_to_anchor = (270, 45, 45, 6))

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(cmap = plt.get_cmap(f'Shifted{cmap_label}')),
            cax = axins, orientation = 'horizontal', ticks = [0,1]
        )
        # axins.xaxis.xticks.horizontalalignment('left')
        axins.set_xticklabels([0.05,0.25])
        axins.tick_params(labelsize = 8)
        axins.xaxis.set_ticks_position('top')
        axins.xaxis.set_label_position('top')
        axins.set_xlabel('Leader fraction', horizontalalignment = 'right', verticalalignment = 'center', fontsize = 7)
        axins.xaxis.set_label_coords(-0.08, .5)

        ax.set_ylabel('$v$')
        ax.set_xlabel('Timestep')
        # ax.set_xlim(-30, 1000)
        # fig.tight_layout()
        fig.savefig(path.join(IMG_DIR, 'optpol_leader_fractions.pdf'))

    def lead_frac_obs_rad():
        data_dir = path.join(DATA_DIR, '20200527', '2-lf_or')
        with open(path.join(data_dir, 'parameters.json')) as f:
            params = json.load(f)

        fig, a = plt.subplots(2,2)
        fig_dict = {
            10: (0,0),
            50: (0,1),
            100: (1,0),
            150: (1,1)
        }

        # for fpath in sorted(glob(path.join(data_dir, '*-v.npy'))):
        #     record_tag = get_rt(fpath)
        #     obs_rad = params[record_tag]['observation_radius']
        #     lead_frac = params[record_tag]['leader_frac']
        #
        #     data = np.load(fpath)
        #     v = [x ** 2 + y ** 2 for (x,y) in data[500:]]
        #     avg_v = sum(v)/len(v)
        #     a[fig_dict[obs_rad]].plot(v, )

        for fpath in sorted(glob(path.join(data_dir, '*-Delta.npy'))):
            record_tag = get_rt(fpath)
            obs_rad = params[record_tag]['observation_radius']
            lead_frac = params[record_tag]['leader_frac']
            delta_t = params[record_tag]['record_every']
            fig_nums = fig_dict[obs_rad]
            plot_delta(a[fig_nums], delta_t)

        for obs_rad, fig_nums in fig_dict.items():
            a[fig_nums].set_title(f'$d = {obs_rad}$')

        plt.tight_layout()
        fig.savefig(path.join(IMG_DIR, 'lead_frac_obs_rad.pdf'))

    def tweaking_the_learning_params():
        fig, a = plt.subplots(2,2)

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

    # def gamma_gradient():
    #     plt.figure()
    #     with open(path.join(DATA_DIR, 'parameters.josn')) as f:
    #         params = json.load()
    #     for dname in sorted(glob(f'{data_dir}/*-Q/')):
    #         record_tag = get_rt(dir)

    # def delta_lf():
    #
    #     for fname in sorted(fnames_gamma):
    #         record_tag = get_rt(fname)
    #         gamma = params[record_tag]['Q_params']['gamma']
    #         Q = np.load(fname)
    #         Delta_l, Delta_f = (0,0)
    #         for i in range(25):
    #             Delta_l += np.sum(Q[i,:,1] - Q[i,:,0] < 0)
    #         for i in range(25, 100):
    #             Delta_f += np.sum(Q[i,:,0] - Q[i,:,1] < 0)
    #         Delta_l /= 25 * Q.shape[1]
    #         Delta_f /= 75 * Q.shape[1]
    #         data.append([gamma, Delta_l, Delta_f])

    def long_run():
        record_tag = '20200503-180806'
        data_dir = path.join(DATA_DIR, '20200501')
        fpath = path.join(data_dir, f'{record_tag}-Delta.npy')
        with open(path.join(data_dir, 'parameters.json')) as f:
            params = json.load(f)
        delta_t = params[record_tag]['record_every']
        plt.figure()
        p.plot_Delta(fpath, record_every = delta_t)

        # Plot Delta_f and Delta_l in the same figure
        fpath = path.join(data_dir, f'{record_tag}-Delta_lf.npy')
        data = np.load(fpath)
        Delta_l = data[:,0]
        Delta_f = data[:,1]
        plt.plot(50_000 * np.arange(len(Delta_l)), Delta_l)
        plt.plot(50_000 * np.arange(len(Delta_f)), Delta_f)

        # Fit results to an exponential curve
        # Delta = np.load(fpath)
        # t = np.arange(0, delta_t * len(Delta), delta_t)
        # p0 = [1e-6]
        # popt, pcov = curve_fit(exp_func, t, Delta, p0 = p0)
        # print(popt)
        # plt.plot(t, exp_func(t, *popt))

        plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (6,6))
        plt.xlabel('Timestep')
        plt.tight_layout()
        plt.savefig(path.join(IMG_DIR, 'long_run.pdf'))

if __name__ == "__main__":
    if sys.argv[-1] == 'update_all' or len(sys.argv) == 1:
        for func in Figures.__dict__.keys():
            if not func.startswith('__'):
                getattr(Figures, func)()
    else:
        getattr(Figures, sys.argv[-1])()
