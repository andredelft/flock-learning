from os import path
import sys
sys.path.append(path.join(sys.path[0], '..', '..'))

from glob import glob
import plot as p
from matplotlib import pyplot as plt

reference_fnames = glob('../20200422/*-Delta.npy')

def gen_fig():
    for fpath in reference_fnames:
        p.plot_Delta(fpath, color = 'Darkgrey')
    p.plot_all(data_dir = '.', quantity = 'Delta')

if __name__ == "__main__":
    plt.figure()
    gen_fig()
    plt.savefig(f'Delta.png', dpi = 300)
