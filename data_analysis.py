import numpy as np
from glob import glob
from os import path, listdir
import json
import regex

from utils import get_rt
from birds import get_maj_obs

BIRD_TYPES = 'fl'
CARD_DIRS = 'NESW'

def card_dir_Delta(data_dir, Q_dirs = []):
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
            timestep = regex.search('\d+', path.split(fpath)[1]).group()
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
