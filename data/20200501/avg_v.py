from os import path
import sys
sys.path.append(path.join(sys.path[0], '..', '..'))

import json
from glob import glob

import plot as p
from main import load_from_Q

def get_tstep(fname):
    return path.splitext(path.split(fname)[1])[0]

def run_sims(fnames, record_tag):
    for fname in fnames:
        load_from_Q(
            fname, data_dir = '.', record_tag = record_tag, sim_length = 1500, 
            record_data = True, plot = False, comment = path.splitext(fname)
        )

def gen_fig(plot_prms, **kwargs):
    for rt, orig in plot_prms:
        tstep = int(get_tstep(orig))
        p.plot_mag(f'{rt}-v.npy', label = f'T = {tstep}')
    plt.xlabel('Timestep')
    plt.ylabel('v')
    plt.legend()
    if expose_remote:
        plt.savefig(
            path.join(path.expanuser('~'), 'public_html', 'avg_v.png'), dpi = 300
        )
    
if __name__ == "__main__":
    record_tag = '20200503-180806'
    Q_files = sorted(glob(f'{record_tag}-Q/*.npy'))
    with open('parameters.json') as f:
        params = json.load(f)
    comments = set(prms.pop('comment', '') for rt, prms in params.items())
    run_sims(
        [fname for fname in Q_files if path.splitext(fname) not in comments], 
        record_tag
    )
    with open('parameters.json') as f:
        # reload params with new runs
        params = json.load(f)
    plot_prms = sorted([
        (rt, prms['comment']) for rt, prms in params.items() 
        if 'comment' in prms.keys() and prms['comment'].startswith(record_tag)
    ], key = prms['comment'])
    gen_fig(plot_prms, expose_remote = True)
