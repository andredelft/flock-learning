from glob import glob
import pickle
import json
from os import path
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append(path.abspath('..'))

from utils import get_rt

def create_data():
    avg_v = {
        '20200521-120133': [],
        '20200521-120138': [],
        '20200521-120143': [],
        '20200521-120148': [],
    }

    with open('orig/parameters.json') as f:
        params = json.load(f)

    for fname in sorted(glob('orig/*-v.npy')):
        record_tag = get_rt(fname)
        comment = params[record_tag]['comment']
        orig_rt = get_rt(comment)
        timestep = int(comment.split('-')[-1])
        data = np.load(fname)
        if len(data) >= 1000:
            v = sum(x**2 + y**2 for (x,y) in data[500:])/len(data[500:])
            avg_v[orig_rt].append((timestep, v))

    with open('avg_v.pickle', 'wb') as f:
        pickle.dump(avg_v, f)

def plot_data():
    with open('avg_v.pickle', 'rb') as f:
        avg_v = pickle.load(f)

    for record_tag, data in avg_v.items():
        timesteps = []
        v = []
        for item in data:
            timesteps.append(item[0])
            v.append(item[1])
        plt.scatter(timesteps, v, marker = '.', label = record_tag)

    plt.legend()
    plt.savefig('avg_v.png', dpi = 300)
