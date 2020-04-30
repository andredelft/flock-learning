import os
import sys
sys.path.append(os.path.join(sys.path[0], '..', '..', '..'))

from glob import glob
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

# reference_fnames = [
#     '../../20200422/20200422-113655-Delta.npy'
# ]
reference_fnames = glob('../../20200422/*-Delta.npy')

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
        p.plot_Delta(fname, color = cmap[par_name](value), label = f'$\\{par_name} = {value}$')
    else:
        p.plot_Delta(fname, color = 'Black', label = record_tag)

def gen_fig(fnames, filter = '', grad_reference = True):

    # Plot the earlier gradient runs as a reference (cf. ../20200416/Delta_grad.py)
    if grad_reference:
        for fpath in reference_fnames:
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

    plt.figure()
    gen_fig(fnames, filter = 'gamma')
    plt.title(f'Tweaking $\\gamma$')
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('$\\Delta$')
    plt.savefig(f'Delta_gamma.png', dpi = 300)

    # plt.figure()
    # gen_fig(fnames, filter = 'gamma', grad_reference = True)
    # plt.legend()
    # plt.title(f'Tweaking $\\gamma$ (new references)')
    # plt.xlabel('Timestep')
    # plt.ylabel('$\\Delta$')
    # plt.savefig('Delta_gamma_new_refs.png', dpi = 300)
