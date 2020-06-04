from matplotlib import pyplot as plt
import numpy as np
from os import path
from glob import glob
import json
import regex

from utils import get_rt

def _parse_fpath(fpath, data_dir):
    record_tag = get_rt(fpath)
    if fpath == record_tag:
        data_dir = 'data'
        fname = ''
    else:
        data_dir, fname = path.split(fpath)
    return data_dir, fname, record_tag

def avg(data, cap = 20):
    return [sum(data[i:i + cap])/cap for i in range(len(data) - cap)]

def plot_mag(fname, cap = 50, max = None, **kwargs):
    data = np.load(fname)
    if max:
        data = data[:max]
    plt.plot(avg([np.linalg.norm(v) for v in data], cap = cap), **kwargs)

def plot_vx(fname, label = '', cap = 50):
    data = np.load(fname)
    plt.plot(avg([v[0] for v in data], cap = cap), label = label)

def plot_arg(fname, label = '', cap = 100):
    data = np.load(fname)
    plt.plot(avg([np.arctan2(v[1],v[0]) for v in data], cap = cap), label = label)

def plot_mag_arg(fname):
    data = np.load(fname)
    fig,a = plt.subplots(2,1)
    a[0].set_title(path.split(fname)[1])
    a[0].plot(avg([np.linalg.norm(v) for v in data], cap = 50))
    a[0].set_ylabel('$|\mathbf{v}|$')
    a[1].set_xlabel('Timestep')
    a[1].set_ylabel('Arg(v)')
    a[1].plot(avg([np.arctan2(v[1],v[0]) for v in data], cap = 100))

def plot_Delta(fname, record_every = 500, **kwargs):
    data = np.load(fname)
    plt.plot(range(0, record_every * len(data), record_every), data, **kwargs)

def plot_delta_lf(fname, **kwargs):
    data_dir = path.split(fname)[0]
    record_tag = get_rt(fname)
    Q_dir = path.join(data_dir, f'{record_tag}-Q')
    with open(path.join(data_dir, f'{record_tag}-instincts.json')) as f:
        instincts = json.load(f)
    no_birds = len(instincts)
    no_leaders = len([inst for inst in instincts if inst == 'E'])
    if path.isdir(Q_dir):
        delta_l = []
        delta_f = []
        timesteps = []
        for Qname in sorted(glob(f'{Q_dir}/*.npy')):
            Q = np.load(Qname)
            print(Q.shape)
            timestep = int(regex.search(r'\d+', path.split(Qname)[1]).group())
            dl = 0
            df = 0
            for i, inst in enumerate(instincts):
                if inst == 'E':
                    dl += np.sum(Q[i,:,1] - Q[i,:,0] < 0)
                else:
                    df += np.sum(Q[i,:,0] - Q[i,:,1] < 0)
            dl /= no_leaders * Q.shape[1]
            df /= (no_birds - no_leaders) * Q.shape[1]
            timesteps.append(timestep)
            delta_l.append(dl)
            delta_f.append(df)
            print(timestep, dl, df)
        plt.plot(timesteps, delta_l, label = f'{record_tag}_l')
        plt.plot(timesteps, delta_f, label = f'{record_tag}_f')
    else:
        print(
            f'The Q-tables of {record_tag} have not been tracked, so delta_l and '
            'delta_f cannot be reproduced'
        )

def plot_all(data_dir = 'data', quantity = 'v', cap = 50, expose_remote = False,
             show_legend = True, title = '', save_as = '', **kwargs):

    if expose_remote or save_as:
        plt.figure()
    with open(path.join(data_dir,'parameters.json')) as f:
        params = json.load(f)

    if quantity == 'Delta_lf':
        quantity = 'Delta'
        lf = True
    elif quantity == 'Delta':
        lf = False

    for i, fname in enumerate(sorted(glob(f'{data_dir}/*-{quantity}.npy'))):
        record_tag = get_rt(fname)
        if quantity == 'v':
            label = record_tag if show_legend else None
            plot_mag(
                fname, cap = cap,
                label = record_tag,
                **kwargs
            )
        elif quantity == 'Delta':
            record_every = params[record_tag].pop('record_every', 500)
            plot_Delta(
                fname, label = record_tag, record_every = record_every,
                **kwargs
            )
            if lf:
                plot_delta_lf(fname, **kwargs)
        elif quantity == 't':
            times = np.load(fname)
            # cum_times = []
            # cum_time = 0
            # for time in times:
            #     cum_time += time
            #     cum_times.append(cum_time)
            tot_time = sum(times)
            comment = params[record_tag].pop('comment', '')
            time = f'{round(tot_time, 2)} s'
            if comment:
                label = f'{comment} ({time})'
            else:
                label = time
            # plt.bar(i, tot_time, label = label)
            plt.plot(times, label = label)

    if quantity == 'v':
        plt.title(f'Magnitude of average velocity vector (Capsize = {cap})')
        plt.ylabel('v')
    elif quantity == 'Delta' or quantity == 'Delta_lf':
        plt.ylabel(r'$\Delta$')
    elif quantity == 't':
        plt.ylabel('Time (sec)')

    plt.xlabel('Timestep')

    if title:
        plt.title(title)

    if show_legend or show_legend == 0:
        if type(show_legend) in [str, int]:
            # Pass in the legend location via legend variable
            plt.legend(loc = show_legend)
        else:
            plt.legend()

    if expose_remote:
        plt.savefig(
            path.join(path.expanduser('~'), f'public_html/{quantity}.png'),
            dpi = 300
        )
    elif save_as:
        plt.savefig(path.join(data_dir, save_as), dpi = 300)

def gen_avg_v(data_dir, save_dir):
    fpaths = sorted(glob(path.join(data_dir, '*-v.npy')))
    avg_v_dict = dict()
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)

    # Collect data and sort by record tag in avg_v_dict
    for fpath in fpaths:
        record_tag = get_rt(fpath)
        comment_parts = params[record_tag].get('comment', '').split('-')
        if len(comment_parts) == 3:
            orig_rt = '-'.join(comment_parts[:2])
            timestep = int(comment_parts[2])
            v = np.load(fpath)
            v_mag = [x ** 2 + y ** 2 for (x,y) in v[500:]]
            if len(v_mag) > 0:
                avg_v = sum(v_mag)/len(v_mag)
                if orig_rt in avg_v_dict.keys():
                    avg_v_dict[orig_rt].append((timestep, avg_v))
                else:
                    avg_v_dict[orig_rt] = [(timestep, timestep)]

    # Reformat data and save
    for record_tag, data in avg_v_dict.items():
        timesteps = []
        avg_v = []
        for entry in data:
            timesteps.append(entry[0])
            avg_v.append(entry[1])
            np.save(
                path.join(save_dir, f'{record_tag}-avg_v.npy'),
                np.array([timesteps, avg_v])
            )
