from os import path
import sys
sys.path.append(path.join(sys.path[0], '..', '..'))

from glob import glob
import plot as p
import utils as u
from matplotlib import pyplot as plt
import numpy as np
import json

reference_fnames = glob('../20200422/*-Delta.npy')
yesterdays_repos = glob('../20200429/*-Delta.npy')

cmap_blue = plt.get_cmap('Blues')
cmap_red = plt.get_cmap('Reds')
repos_cmap_dict = {
        0: 0.25,
     1000: 0.50,
     5000: 0.75,
    10000: 0.90
}

def gen_fig():
    for fpath in reference_fnames:
        p.plot_Delta(fpath, color = 'Darkgrey')
    for fpath in yesterdays_repos:
        p.plot_Delta(fpath, color = cmap_red(0.7))
    with open('parameters.json') as f:
        params = json.load(f)
    for fpath in glob('*-Delta.npy'):
        record_tag = u.get_rt(fpath)
        repos_every = params[record_tag].pop('repos_every', 0)
        p.plot_Delta(
            fpath, color = cmap_blue(repos_cmap_dict[repos_every]),
            label = f'repos_every = {repos_every}'
        )

if __name__ == "__main__":
    plt.figure()
    gen_fig()
    plt.xlabel('Timestep')
    plt.ylabel('$\\Delta$')
    plt.savefig(f'Delta.png', dpi = 300)
