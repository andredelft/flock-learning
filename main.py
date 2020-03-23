from field import Field
from os import path
import json

def load_from_Q(record_tag, data_dir = 'data'):
    with open(path.join(data_dir, 'parameters.json')) as f:
        params = json.load(f)[record_tag]
    no_birds = params.pop('no_birds')
    Q_file = path.join(data_dir, f'{record_tag}-Q.npy')
    params.pop('learning_alg')
    print(params)
    Field(
        no_birds, plot = True, record_data = True, learning_alg = 'pol_from_Q',
        Q_file = Q_file, comment = record_tag, **params
    )

if __name__ == '__main__':
    load_from_Q('20200311-150909', data_dir = 'data/20200311')
    # Field(
    #     100, plot = True, record_data = False, observe_direction = True,
    #     sim_length = 12500, learning_alg = 'pol_from_Q', Q_file = 'data/20200311/20200311-150909-Q.npy'
    # )
