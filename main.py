from field import Field
from os import path
import json

def load_from_Q(record_tag, data_dir = 'data'):
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)[record_tag]
    no_birds = params.pop('no_birds')
    Q_file = path.join(data_dir, f'{record_tag}-Q.npy')
    params.pop('learning_alg')
    if 'Q_params' in params.keys():
        params.update(params.pop('Q_params'))
    print(params)
    Field(
        no_birds, plot = True, record_data = True, learning_alg = 'pol_from_Q',
        Q_file = Q_file, comment = record_tag, **params
    )

if __name__ == '__main__':
    load_from_Q('20200323-172709', data_dir = 'data/20200323/2-Larger_Epsilon')
    # Field(
    #     100, plot = True, record_data = True, observe_direction = True,
    #     alpha = 0.9, gamma = 0.9, epsilon = 0.9, learning_alg = 'Q', sim_length = 12500
    # )
