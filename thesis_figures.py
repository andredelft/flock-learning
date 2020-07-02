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
# USER_ROOT = path.expanduser('~')
# IMG_DIR = path.join(USER_ROOT, 'public_html')
IMG_DIR = path.join(HERE, 'thesis-figures')

FIGSIZE_X, FIGSIZE_Y = [5,4]

mpl.rcParams['figure.figsize'] = [FIGSIZE_X, FIGSIZE_Y]
mpl.rcParams['font.family'] = 'serif'

# Use tex and match style to thesis
mpl.rc('text', usetex = True)
mpl.rcParams['text.latex.preamble']=[r'\usepackage{amsmath,palatino,mathpazo}']

# Register some shifted colormaps
VMIN, VMAX = (0.2, 1)

for name in ['Blues', 'Reds', 'Greens']:
    shiftedColorMap(
        plt.get_cmap(name), VMIN, (VMAX + VMIN)/2, VMAX, f'Shifted{name}'
    )

LP_CMAPS = {
    'alpha': plt.get_cmap('ShiftedBlues'),
    'gamma': plt.get_cmap('ShiftedReds'),
    'epsilon': plt.get_cmap('ShiftedGreens')
    # 'references': plt.get_cmap('Greys')
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

ORIG_CMAP_LABEL = 'YlOrRd'
LF_VMIN, LF_VMAX = (0.15, 0.9)
LF_CMAP_LABEL = f'Shifted{ORIG_CMAP_LABEL}'
shiftedColorMap(
    plt.get_cmap(ORIG_CMAP_LABEL), LF_VMIN, (LF_VMAX + LF_VMIN)/2, LF_VMAX, LF_CMAP_LABEL
)
LF_CMAP = plt.get_cmap(LF_CMAP_LABEL)

# The figures:

__all__ = [
    'v_delta', 'optpol_leader_fractions', 'learning_params',
    'lead_frac_obs_rad_discrete', 'lead_frac_obs_rad_gradient',
    'gamma_long_term'
]

def v_delta():
    fig,a = plt.subplots(2,1)
    dir1 = path.join(DATA_DIR, '20200424', '2-avg_v')
    dir2 = path.join(DATA_DIR, '20200425')
    data = []
    data.append(np.load(path.join(dir1, 'avg_v.npy')))
    data.append(np.load(path.join(dir2, 'avg_v.npy')))

    for i,d in enumerate(data):
        a[i].scatter(*d, marker = '.')
        a[i].set_ylim(0, 1.05)
        a[i].set_ylabel(r'$\langle v \rangle$')
    a[1].set_xlabel(r'$\Delta$')
    a[0].set_xlim(0,1)
    a[1].set_xlim(0,0.5)
    fig.tight_layout()
    fig.savefig(path.join(IMG_DIR, 'v_delta.pdf'))

def optpol_leader_fractions():
    fig, a = plt.subplots(1,2, figsize = [FIGSIZE_X, 0.55 * FIGSIZE_Y])
    data_dir = path.join(DATA_DIR, '20200529', '1-opt_pol')
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)
    # record_tags = [rt for rt in params.keys() if not rt.startswith('desired')]

    for fpath in sorted(glob(f'{data_dir}/*-v.npy')):
        record_tag = get_rt(fpath)
        leader_frac = params[record_tag]['leader_frac']
        color = LF_CMAP(leader_frac/0.25)
        v = np.load(fpath)[:4000]
        v_mag = p.avg([x ** 2 + y ** 2 for (x,y) in v], cap = 50)
        v_arg = [np.arctan2(y,x) for (x,y) in v]
        a[0].plot(v_mag, c = color)
        a[1].plot(v_arg, c = color)

    a[0].set_ylabel(r'$|\boldsymbol{v}|$')
    a[1].set_ylabel(r'$\arg(\boldsymbol{v})$')
    a[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    a[1].set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$', r'$\pi$'])
    # a[1].yaxis.set_label_position('right')
    # a[1].yaxis.set_ticks_position('right')
    for i in range(2):
        a[i].set_xlabel('Timestep')
        a[i].ticklabel_format(style = 'sci', axis = 'x', scilimits = (3,3))

    fig.tight_layout()

    axin = inset_axes(a[0], width = '100%', height = '100%', bbox_to_anchor = (118, 53, 45, 6))

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(cmap = LF_CMAP),
        cax = axin, orientation = 'horizontal', ticks = [0,1]
    )
    # axins.xaxis.xticks.horizontalalignment('left')
    axin.set_xticklabels([0, 0.25])
    axin.tick_params(labelsize = 8)
    axin.xaxis.set_ticks_position('top')
    axin.xaxis.set_label_position('top')
    axin.set_xlabel(r'$l$', horizontalalignment = 'right', verticalalignment = 'center', fontsize = 9)
    axin.xaxis.set_label_coords(-0.08, .5)

    fig.savefig(path.join(IMG_DIR, 'optpol_leader_fractions.pdf'))

def learning_params():
    figsize_x = 1.1 * FIGSIZE_X
    figsize_y = 1.5 * FIGSIZE_Y
    fig, a = plt.subplots(3, 2, figsize = [figsize_x, figsize_y])

    pars = ['alpha', 'gamma', 'epsilon']
    record_every = 100

    delta_dir = path.join(DATA_DIR, '20200604', '1-lp_data')
    avg_v_dir = path.join(DATA_DIR, '20200604', '2-avg_v')

    with open(path.join(delta_dir, 'parameters.json')) as f:
        params = json.load(f)

    # All runs together is a bit much, thus only a few are allowed.
    # This can be configured below:
    allowed_inds = [
        0,  # 1 (0.99 for gamma)
        # 1,  # 0.9
        2,  # 0.8
        # 3,  # 0.7
        4,  # 0.6
        # 5,  # 0.5
        6,  # 0.4
        # 7,  # 0.3
        8,  # 0.2
        9,  # 0.1
        10, # 0.0
    ]

    for i,par in enumerate(pars):
        record_tags = [rt for rt in params if params[rt]['comment'] == f'vary_{par}']
        # sort by increasing parameter value
        record_tags.sort(key = lambda rt: params[rt]['Q_params'][par], reverse = True)

        for rt_ind in allowed_inds:
            record_tag = record_tags[rt_ind]
            par_value = params[record_tag]['Q_params'][par]
            plot_delta(
                a[i,0], f'{record_tag}-Delta.npy', data_dir = delta_dir,
                delta_t = record_every, color = LP_CMAPS[par](par_value)
            )
            fpath = path.join(avg_v_dir, f'{record_tag}-avg_v.npy')
            if path.isfile(fpath):
                data = np.load(fpath)
                a[i, 1].scatter(*data, color = LP_CMAPS[par](par_value), marker = '.')

    for i in range(3):
        a[i, 0].set_ylim(0.471, 0.503)
        a[i, 0].set_ylabel(r'$\Delta$')
        a[i, 1].set_ylabel(r'$\langle v \rangle$', rotation = 270, labelpad = 17)
        a[i, 1].yaxis.set_label_position('right')
        a[i, 1].yaxis.set_ticks_position('right')
        if i != 2:
            for j in range(2):
                a[i,j].set_xticklabels([])
        else:
            for j in range(2):
                a[i,j].set_xlabel('Timestep')
                a[i,j].ticklabel_format(style = 'sci', axis = 'x', scilimits = (3,3))

    fig.tight_layout()

    axins = dict()
    for i, par in enumerate(['alpha', 'gamma', 'epsilon']):
        axins[par] = inset_axes(a[i, 0], width='100%', height='100%', bbox_to_anchor = (65, 53 + 131 * (2 - i), 45, 6))
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(cmap = LP_CMAPS[par]), cax=axins[par],
            orientation='horizontal', ticks=[0, 1]
        )
        cbar.ax.set_ylabel(fr'$\{par}$  ', rotation = 0, x = -0.5, y = -0.4)
        axins[par].xaxis.set_ticks_position('top')
        axins[par].xaxis.set_label_position('top')

    axins['gamma'].set_xticklabels([0, 0.99])

    fig.savefig(path.join(IMG_DIR,'learning_params.pdf'))

def lead_frac_obs_rad_discrete(
    data_dir_left = path.join(DATA_DIR, '20200528', '1-lf_or'),
    data_dir_right = path.join(DATA_DIR, '20200528', '2-avg_v'),
    save_as = 'lead_frac_obs_rad_discrete.pdf'):

    figsize_x = 1.1 * FIGSIZE_X
    figsize_y = 2 * FIGSIZE_Y
    fig, a = plt.subplots(4, 2, figsize = [figsize_x, figsize_y])

    or_dict = {
         10: 0,
         50: 1,
        100: 2,
        150: 3
    }

    data_dir = data_dir_left
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)

    for fpath in sorted(glob(path.join(data_dir, '*-Delta.npy'))):
        record_tag = get_rt(fpath)
        obs_rad = params[record_tag]['observation_radius']
        lead_frac = params[record_tag]['leader_frac']
        delta_t = params[record_tag]['record_every']
        fig_nums = (or_dict[obs_rad], 0)
        color = LF_CMAP(params[record_tag]['leader_frac']/0.40)
        # label = r'$\Delta$' if obs_rad == 50 and lead_frac == 0.40 else None
        plot_delta(a[fig_nums], fpath, delta_t = delta_t, c = color)

    data_dir = data_dir_right

    for fpath in sorted(glob(path.join(data_dir, '*-avg_v.npy'))):
        record_tag = get_rt(fpath)
        obs_rad = params[record_tag]['observation_radius']
        lead_frac = params[record_tag]['leader_frac']
        fig_nums = (or_dict[obs_rad], 1)
        data = np.load(fpath)
        color = LF_CMAP(params[record_tag]['leader_frac']/0.40)
        # label = r'$\langle v \rangle$' if obs_rad == 50 and lead_frac == 0.40 else None
        a[fig_nums].scatter(data[0], data[1], marker = '.', c = len(data[0]) * [color])

    for obs_rad, fig_row in or_dict.items():
        a[fig_row, 0].set_ylim(0.47, 0.51)
        a[fig_row, 0].set_title(f'$d/L = {obs_rad/800}$', loc = 'left')
        n = round(np.pi * obs_rad**2 * 100 / (800**2), 2)
        a[fig_row, 1].set_title(f'$n = {n}$', loc = 'right')
        a[fig_row, 1].set_ylim(-0.05, 1.05)
        a[fig_row, 0].set_ylabel(r'$\Delta$')
        a[fig_row, 1].set_ylabel(r'$\langle v \rangle$', rotation = 270, labelpad = 17)
        a[fig_row, 1].yaxis.set_label_position('right')
        a[fig_row, 1].yaxis.set_ticks_position('right')

    for j in range(2):
        a[3,j].set_xlabel('Timestep')
        a[3,j].ticklabel_format(style = 'sci', axis = 'x', scilimits = (3,3))
        for i in range(3):
            a[i,j].set_xticklabels([])

    fig.tight_layout()

    # axin = inset_axes(a[0,0], width = '100%', height = '100%', bbox_to_anchor = (143, 2 * 185, 35, 5))
    axin = inset_axes(a[0,1], width='30%', height='5%', loc='upper center')
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(cmap = LF_CMAP),
        cax = axin, orientation = 'horizontal', ticks = [0,1]
    )
    axin.set_xticklabels([0, 0.40])
    axin.tick_params(labelsize = 8)
    axin.set_xlabel(r'$l$', horizontalalignment = 'right', verticalalignment = 'center')
    axin.xaxis.set_label_coords(-0.08, .5)

    fig.savefig(path.join(IMG_DIR, save_as))

def lead_frac_obs_rad_gradient():
    # Same figure, different data
    lead_frac_obs_rad_discrete(
        data_dir_left = path.join(DATA_DIR, '20200527', '2-lf_or'),
        data_dir_right = path.join(DATA_DIR, '20200527', '3-avg_v'),
        save_as = 'lead_frac_obs_rad_gradient.pdf'
    )


def gamma_long_term():
    fig, a = plt.subplots(4, 2, figsize = [FIGSIZE_X, 1.9 * FIGSIZE_Y])
    fig_norm, a_norm = plt.subplots(2, 2, figsize = [FIGSIZE_X, 0.95 * FIGSIZE_Y])
    data_dir = path.join(DATA_DIR, '20200612')
    bird_types = 'lf'
    card_dirs = 'NESW'

    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)

    record_tags = [
        rt for rt in params.keys() if params[rt]['comment'] == 'vary_gamma'
    ]

    for record_tag in record_tags:
        gamma = params[record_tag]['Q_params']['gamma']
        for i, card_dir in enumerate(card_dirs):
            for j, bird_type in enumerate(bird_types):
                fname = f'{record_tag}-Delta_{bird_type}{card_dir}.npy'
                fpath = path.join(data_dir, fname)
                if path.isfile(fpath):
                    data = np.load(path.join(data_dir, fname))
                    # data -= data[1,0]
                    a[i,j].plot(*data, color = LP_CMAPS['gamma'](gamma))
                    if j == 1:
                        # data += 0.5 - data[1,0]
                        data -= data[1,0]
                        fignum_norm = tuple(int(n) for n in np.binary_repr(i, width = 2))
                        a_norm[fignum_norm].plot(*data, color = LP_CMAPS['gamma'](gamma))

    a[0,0].set_title('Leaders')
    a[0,1].set_title('Followers')

    for i, card_dir in enumerate(card_dirs):
        a[i,0].set_ylabel(rf'$\Delta_\mathsf{{{card_dir}}}$')
        a[i,0].set_ylim([0.251, 0.52])
        # a[i,1].set_yticklabels([])
        for j, bird_type in enumerate(bird_types):
            if i != 3:
                a[i,j].set_xticklabels([])
            else:
                a[i,j].set_xlabel('Timestep')
                a[i,j].ticklabel_format(style = 'sci', axis = 'x', scilimits = (3,3))

        fignum_norm = tuple(int(n) for n in np.binary_repr(i, width = 2))
        a_norm[fignum_norm].set_ylabel(
            rf'$\Delta_\mathsf{{{card_dir}}}(t) - \Delta_\mathsf{{{card_dir}}}(0)$'
        )
        if fignum_norm[0] == 0:
            a_norm[fignum_norm].set_xticklabels([])
        else:
            a_norm[fignum_norm].ticklabel_format(style = 'sci', axis = 'x', scilimits = (3,3))
            a_norm[fignum_norm].set_xlabel('Timestep')

    for i in [0,2]:
        a[i,1].set_ylim([0.442, 0.5052])
    a[1,1].set_ylim([0.251, 0.52])
    # a[3,1].set_ylim([0.489, 0.5042])

    fig.tight_layout()
    fig_norm.tight_layout()

    axin = inset_axes(a[0, 0], width='100%', height='100%', bbox_to_anchor = (124, 515, 45, 6))
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap = LP_CMAPS['gamma']), cax = axin,
        orientation='horizontal', ticks=[0, 1]
    )
    cbar.ax.set_ylabel(fr'$\gamma$  ', rotation = 0, x = -0.5, y = -0.4)
    # axin.xaxis.set_ticks_position('top')
    # axin.xaxis.set_label_position('top')

    axin.set_xticklabels([0, 0.99])

    fig.savefig(path.join(IMG_DIR, 'gamma_long_term.pdf'))
    fig_norm.savefig(path.join(IMG_DIR, 'gamma_long_term_normalized.pdf'))

if __name__ == "__main__":
    if sys.argv[-1] == 'update_all' or len(sys.argv) == 1:
        for func in __all__:
            print(f'Generating {func}.pdf')
            eval(func)()
    else:
        eval(sys.argv[-1])()
