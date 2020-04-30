from os import path
import sys
sys.path.append(path.join(sys.path[0], '..', '..', '..'))

from matplotlib import pyplot as plt
from glob import glob

import plot as p
import utils as u

labels = {
    '20200430-121810': 'LOC1',
    '20200430-124254': 'LOC5',
    '20200430-121957': 'REM1',
    '20200430-124148': 'REM5'
}

def gen_fig():
    for fname in glob('../*-Delta.npy') + glob('../../20200429/*-Delta.npy'):
        p.plot_Delta(fname, color = 'Darkgrey')
    for fname in sorted(glob('*-Delta.npy')):
        record_tag = u.get_rt(fname)
        p.plot_Delta(fname, label = labels[record_tag])

if __name__ == "__main__":
    plt.figure()
    gen_fig()
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('$\\Delta$')
    plt.savefig(f'Delta.png', dpi = 300)
