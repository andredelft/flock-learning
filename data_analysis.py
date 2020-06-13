import numpy as np
from glob import glob
from os import path, listdir
import json
import regex

from utils import get_rt
from birds import get_maj_obs

def avg_v(data_dir, save_dir):
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

BIRD_TYPES = 'fl'
CARD_DIRS = 'NESW'

def Delta_card_dirs(data_dir, Q_dirs = []):
    if not Q_dirs:
        Q_dirs = [dname for dname in listdir(data_dir) if dname.endswith('-Q')]

    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)

    maj = get_maj_obs(print_sizes = False)

    for Q_dir in Q_dirs:
        record_tag = get_rt(Q_dir)
        timesteps = []

        Delta_all = {
            bird_type: {card_dir: [] for card_dir in CARD_DIRS}
            for bird_type in BIRD_TYPES
        }

        with open(path.join(data_dir, f'{record_tag}-instincts.json')) as f:
            instincts = json.load(f)

        bird_inds = {bird_type: [] for bird_type in BIRD_TYPES}
        for i, inst in enumerate(instincts):
            if inst == 'E':
                bird_inds['l'].append(i)
            else:
                bird_inds['f'].append(i)

        for fpath in sorted(glob(path.join(data_dir, Q_dir, '*.npy'))):
            print(fpath)
            timestep = int(regex.search('\d+', path.split(fpath)[1]).group())
            Q = np.load(fpath)
            for card_dir, maj_obs in maj.items():
                for pref_ind, bird_type in enumerate(BIRD_TYPES):
                    Delta = 0
                    for i in bird_inds[bird_type]:
                        for o in maj_obs:
                            if Q[i,o,pref_ind] < Q[i,o,1-pref_ind]:
                                Delta += 1
                    Delta /= len(bird_inds[bird_type]) * len(maj_obs)
                    Delta_all[bird_type][card_dir].append(Delta)
            timesteps.append(timestep)

        for bird_type in BIRD_TYPES:
            for card_dir in CARD_DIRS:
                fname = f'{record_tag}-Delta_{bird_type}{card_dir}.npy'
                np.save(
                    path.join(data_dir, fname),
                    [timesteps, Delta_all[bird_type][card_dir]]
                )

        print(f'Delta files for {record_tag} saved')
