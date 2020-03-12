from matplotlib import pyplot as plt
import numpy as np
from os import path
from glob import glob
import json
import pickle
import regex

from birds import discrete_Vicsek

re_tag = regex.compile(r'^[0-9]+-[0-9]+')

def find_tag(fpath):
    fname = path.split(fpath)[1]
    return re_tag.search(fname).group()

def avg(data, cap = 20):
    return [sum(data[i:i + cap])/cap for i in range(len(data) - cap)]

def plot_mag(fname, label = '', cap = 50):
    data = np.load(fname)
    plt.plot(avg([np.linalg.norm(v) for v in data], cap=cap),label=label)

def plot_vx(fname, label = '', cap = 50):
    data = np.load(fname)
    plt.plot(avg([v[0] for v in data], cap=cap),label=label)

def plot_arg(fname, label = '', cap = 100):
    data = np.load(fname)
    plt.plot(avg([np.arctan2(v[1],v[0]) for v in data], cap=cap),label=label)

def plot_mag_arg(fname):
    data = np.load(fname)
    fig,a = plt.subplots(2,1)
    a[0].set_title(path.split(fname)[1])
    a[0].plot(avg([np.linalg.norm(v) for v in data], cap=50))
    a[0].set_ylabel('$|\mathbf{v}|$')
    a[1].set_xlabel('Timestep')
    a[1].set_ylabel('Arg(v)')
    a[1].plot(avg([np.arctan2(v[1],v[0]) for v in data], cap=100))

maj_obs = {
    'N': [],
    'E': [],
    'S': [],
    'W': []
}

# Calculate Vicsek results
for i in range(3**4):
    tern = f'{np.base_repr(i, base=3):0>4}'
    dir = discrete_Vicsek({d: int(n) for d,n in zip('NESW',tern)})
    if dir != 0:
        maj_obs[dir].append(i)

def avg_pol(fname, no_leaders=25, no_birds=100, Q=False):
    # NB: if Q == True, Q-values are loaded instead of policies
    # (so average leader Q-values and follower Q-values are returned in this case).
    last_pol = np.load(fname)
    avg_leader_pol = {}
    avg_follower_pol = {}
    for dir, inds in maj_obs.items():
        leader = sum(
            sum(l[i] for i in inds)/len(inds)
            for l in last_pol[:no_leaders]
        )/no_leaders
        follower = sum(
            sum(o[i] for i in inds)/len(inds)
            for o in last_pol[no_leaders:no_birds]
        )/(no_birds - no_leaders)
        avg_leader_pol[dir] = leader
        avg_follower_pol[dir] = follower
    return avg_leader_pol, avg_follower_pol

def plot_hist(fpath, plot_policies = True):
    record_tag = find_tag(fpath)
    if fpath == record_tag:
        data_dir = 'data'
    else:
        data_dir, _ = path.split(fpath)

    # Extract necessary parameters
    with open(path.join(data_dir,'parameters.json')) as f:
        params = json.load(f)[record_tag]
    no_birds = params['no_birds']
    leader_frac = params['leader_frac']
    no_leaders = int(no_birds * leader_frac)
    A = params['action_space']
    Q = True if params['learning_alg'] == 'Q' else False

    if Q:
        fname = f'{record_tag}-Q.npy'
    else: # assume learning_alg == 'Ried'
        fname = f'{record_tag}-policies.npy'

    avg_leader_pol, avg_follower_pol = avg_pol(
        path.join(data_dir,fname), no_birds = no_birds, no_leaders = no_leaders, Q = Q
    )

    if not plot_policies:
        with open(path.join(data_dir, f'{record_tag}-instincts.json')) as f:
            instincts = json.load(f)

        N,S,W = [len([i for i in instincts if i == dir]) for dir in 'NSW']
        tot = sum([N,S,W])
        N /= tot
        S /= tot
        W /= tot

    width = 0.35
    fig,a = plt.subplots(2,2)

    if plot_policies:
        labels = A
        x = np.arange(len(labels))
        for i, dir in enumerate(['N','E','S','W']):
            bin = tuple(int(n) for n in f"{i:02b}")
            a[bin].bar(x - width/2, np.array(avg_leader_pol[dir])/sum(avg_leader_pol[dir]), width, label='Leaders')
            a[bin].bar(x + width/2, np.array(avg_follower_pol[dir])/sum(avg_follower_pol[dir]), width, label='Followers')
            a[bin].set_title(f'Majority {dir}')
            a[bin].set_ylim(0,1.05)
            a[bin].set_xticks(x)
            a[bin].set_xticklabels(labels)
    else:
        labels = ['N', 'E', 'S', 'W']
        x = np.arange(len(labels))
        if A == ['V','I']:
            # Majority North
            l_hist = [avg_leader_pol['N'][0],avg_leader_pol['N'][1],0,0]
            f_vicsek = avg_follower_pol['N'][1]
            f_instinct = avg_follower_pol['N'][1]
            f_hist = [f_vicsek + f_instinct * N, 0, f_instinct * S, f_instinct * W]

            a[0,0].bar(x - width/2, l_hist, width, label='Leaders')
            a[0,0].bar(x + width/2, f_hist, width, label='Followers')
            a[0,0].set_title('Majority North')
            a[0,0].set_ylim(0,1.05)
            a[0,0].set_xticks(x)
            a[0,0].set_xticklabels(labels)

            # Majority East
            l_hist = [0,sum(avg_leader_pol['E']),0,0]
            f_vicsek = avg_follower_pol['E'][0]
            f_instinct = avg_follower_pol['E'][1]
            f_hist = [f_instinct * N, f_vicsek, f_instinct * S, f_instinct * W]

            a[0,1].bar(x - width/2, l_hist, width, label='Leaders')
            a[0,1].bar(x + width/2, f_hist, width, label='Followers')
            a[0,1].set_title('Majority East')
            a[0,1].set_ylim(0,1.05)
            a[0,1].set_xticks(x)
            a[0,1].set_xticklabels(labels)
            a[0,1].legend()

            # Majority South
            l_hist = [0,avg_leader_pol['S'][1],avg_leader_pol['S'][0],0]
            f_vicsek = avg_follower_pol['S'][1]
            f_instinct = avg_follower_pol['S'][1]
            f_hist = [f_instinct * N, 0, f_vicsek + f_instinct * S, f_instinct * W]

            a[1,0].bar(x - width/2, l_hist, width, label='Leaders')
            a[1,0].bar(x + width/2, f_hist, width, label='Followers')
            a[1,0].set_title('Majority South')
            a[1,0].set_ylim(0,1.05)
            a[1,0].set_xticks(x)
            a[1,0].set_xticklabels(labels)

            # Majority West
            l_hist = [0,avg_leader_pol['W'][1],0,avg_leader_pol['W'][0]]
            f_vicsek = avg_follower_pol['W'][1]
            f_instinct = avg_follower_pol['W'][1]
            f_hist = [f_instinct * N, 0, f_instinct * S, f_vicsek + f_instinct * W]

            a[1,1].bar(x - width/2, l_hist, width, label='Leaders')
            a[1,1].bar(x + width/2, f_hist, width, label='Followers')
            a[1,1].set_title('Majority West')
            a[1,1].set_ylim(0,1.05)
            a[1,1].set_xticks(x)
            a[1,1].set_xticklabels(labels)

        elif A == ['N','E','S','W','I']:
            # Majority North
            l_hist = [
                avg_leader_pol['N'][0], avg_leader_pol['N'][1] + avg_leader_pol['N'][4],
                avg_leader_pol['N'][2], avg_leader_pol['N'][3]
            ]
            f_inst = avg_follower_pol['N'][4]
            f_hist = [
                avg_follower_pol['N'][0] + f_inst * N, avg_follower_pol['N'][1],
                avg_follower_pol['N'][2] + f_inst * S, avg_follower_pol['N'][3] + f_inst * W
            ]

            a[0,0].bar(x - width/2, l_hist, width, label='Leaders')
            a[0,0].bar(x + width/2, f_hist, width, label='Followers')
            a[0,0].set_title('Majority North')
            a[0,0].set_ylim(0,1.05)
            a[0,0].set_xticks(x)
            a[0,0].set_xticklabels(labels)

            # Majority East
            l_hist = [
                avg_leader_pol['E'][0], avg_leader_pol['E'][1] + avg_leader_pol['E'][4],
                avg_leader_pol['E'][2], avg_leader_pol['E'][3]
            ]
            f_inst = avg_follower_pol['E'][4]
            f_hist = [
                avg_follower_pol['E'][0] + f_inst * N, avg_follower_pol['E'][1],
                avg_follower_pol['E'][2] + f_inst * S, avg_follower_pol['E'][3] + f_inst * W
            ]

            a[0,1].bar(x - width/2, l_hist, width, label='Leaders')
            a[0,1].bar(x + width/2, f_hist, width, label='Followers')
            a[0,1].set_title('Majority East')
            a[0,1].set_ylim(0,1.05)
            a[0,1].set_xticks(x)
            a[0,1].set_xticklabels(labels)
            a[0,1].legend()

            # Majority South
            l_hist = [
                avg_leader_pol['S'][0], avg_leader_pol['S'][1] + avg_leader_pol['S'][4],
                avg_leader_pol['S'][2], avg_leader_pol['S'][3]
            ]
            f_inst = avg_follower_pol['S'][4]
            f_hist = [
                avg_follower_pol['S'][0] + f_inst * N, avg_follower_pol['S'][1],
                avg_follower_pol['S'][2] + f_inst * S, avg_follower_pol['S'][3] + f_inst * W
            ]

            a[1,0].bar(x - width/2, l_hist, width, label='Leaders')
            a[1,0].bar(x + width/2, f_hist, width, label='Followers')
            a[1,0].set_title('Majority South')
            a[1,0].set_ylim(0,1.05)
            a[1,0].set_xticks(x)
            a[1,0].set_xticklabels(labels)

            # Majority West
            l_hist = [
                avg_leader_pol['W'][0], avg_leader_pol['W'][1] + avg_leader_pol['W'][4],
                avg_leader_pol['W'][2], avg_leader_pol['W'][3]
            ]
            f_inst = avg_follower_pol['W'][4]
            f_hist = [
                avg_follower_pol['W'][0] + f_inst * N, avg_follower_pol['W'][1],
                avg_follower_pol['W'][2] + f_inst * S, avg_follower_pol['W'][3] + f_inst * W
            ]

            a[1,1].bar(x - width/2, l_hist, width, label='Leaders')
            a[1,1].bar(x + width/2, f_hist, width, label='Followers')
            a[1,1].set_title('Majority West')
            a[1,1].set_ylim(0,1.05)
            a[1,1].set_xticks(x)
            a[1,1].set_xticklabels(labels)

    st = fig.suptitle(
        f'Average final Q-values of agents in {"".join(A)}-model', fontsize = 'x-large'
    )

    fig.tight_layout()

    # Shift plots downward so that suptitle doesn't interfere with subplots titles
    st.set_y(0.97)
    fig.subplots_adjust(top=0.85)

def plot_all(data_dir = 'data', quantity = 'mag', cap = 50):
    with open(path.join(data_dir,'parameters.json')) as f:
        pars = json.load(f)
    for fname in sorted(glob(f'{data_dir}/*-v.npy')):
        record_tag = find_tag(fname)
        if quantity == 'mag':
            plot_mag(
                fname, cap = cap,
                label = record_tag
            )
    if quantity == 'mag':
        plt.title(f'Magnitude of average velocity vector (Capsize = {cap})')
        plt.ylabel('v')
    plt.xlabel('Timestep')
    plt.legend()
