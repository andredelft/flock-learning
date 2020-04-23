import os
import sys
sys.path.append(os.path.join(sys.path[0], '..', '..'))

import plot as p
from matplotlib import pyplot as plt
import numpy as np
import json

cmap = {
    'alpha': plt.get_cmap('Blues'),
    'gamma': plt.get_cmap('Reds'),
    'epsilon': plt.get_cmap('Greens'),
    # 'references': plt.get_cmap('Greys')
}

gradient_fnames = [
    # '../20200418/20200418-114214-Delta.npy',
    '../20200417/20200417-131821-Delta.npy',
    '../20200416/20200416-190443-Delta.npy',
    '../20200415/20200415-144158-Delta.npy',
    '../20200415/20200415-154211-Delta.npy',
    '../20200416/20200416-092043-Delta.npy',
    '../20200416/20200416-125734-Delta.npy',
    '../20200415/20200415-122210-Delta.npy',
    '../20200416/20200416-114526-Delta.npy'
]

with open('parameters.json') as f:
    params = json.load(f)

def get_rt(fname):
    return '-'.join(fname.split('-')[:2])

def get_lp(record_tag, paramas = params):
    if 'comment' in params[record_tag].keys():
        par_name = params[record_tag]['comment'].split('_')[1]
        value = params[record_tag]['Q_params'][par_name]
        return par_name, value
    else: # Run has the same parameters as usual (thus should be comparable to earlier gradient runs)
        return '', 0

def plot_lp(fname, par_name = '', value = 0):
    record_tag = get_rt(fname)
    if par_name != '':
        p.plot_Delta(fname, color = cmap[par_name](0.75 * value + 0.25), label = f'$\\{par_name} = {value}$')
    else:
        p.plot_Delta(fname, color = 'Black', label = record_tag)

def gen_fig(fnames, filter = '', grad_reference = True):

    # Plot the earlier gradient runs as a reference (cf. ../20200416/Delta_grad.py)
    if grad_reference:
        for fpath in gradient_fnames:
            # data_dir, fname = os.path.split(fpath)
            # record_tag = get_rt(fname)
            # with open(os.path.join(data_dir, 'parameters.json')) as f:
            #     R = json.load(f)[record_tag]['reward_signal']
            p.plot_Delta(fpath, color = 'Darkgrey')

    record_tags = [get_rt(fname) for fname in fnames]
    learning_pars = [get_lp(record_tag) for record_tag in record_tags]

    if filter:
        allowed_inds = sorted(
            (i for i, lp in enumerate(learning_pars) if lp[0] == filter),
            key = lambda i: learning_pars[i][1]
        )
    else:
        allowed_inds = range(len(fnames))

    for i in allowed_inds:
        plot_lp(fnames[i], *learning_pars[i])


if __name__ == "__main__":

    fnames = sorted(fname for fname in os.listdir() if fname.endswith('-Delta.npy'))
    record_tags = [get_rt(fname) for fname in fnames]
    learning_pars = [get_lp(record_tag) for record_tag in record_tags]

    for par in ['', 'alpha', 'gamma', 'epsilon']:
        plt.figure()
        gen_fig(fnames, filter = par)
        plt.title(f'Tweaking $\\{filter}$')
        plt.legend()
        plt.xlabel('Timestep')
        plt.ylabel('$\\Delta$')
        # plt.savefig(f'Delta_{par if par else "all"}.png', dpi = 300)

    plt.figure()
    gen_fig(fnames, filter = 'gamma', grad_reference = True)
    p.plot_all(data_dir = '../20200423/new_refs', quantity = 'Delta', color = 'dimgray')
    plot_lp('20200422-152117-Delta.npy')
    plot_lp('20200422-160846-Delta.npy')
    plt.legend()
    plt.title(f'Tweaking $\\gamma$ (new references)')
    plt.xlabel('Timestep')
    plt.ylabel('$\\Delta$')
    plt.savefig('Delta_gamma_new_refs.png', dpi = 300)
