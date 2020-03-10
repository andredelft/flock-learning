from field import Field

if __name__ == '__main__':
    for alpha in [0.9,0.7,0.5,0.3,0.1]:
        Field(
            100, periodic = True, plot = False, record_data = True, observe_direction = True,
            track_birds = False, sim_length = 5000, learning_alg = 'Q', comment = 'vary_alpha',
            alpha = alpha, gamma = 0.9
        )
    for gamma in [0.9,0.7,0.5,0.3,0.1]:
        Field(
            100, periodic = True, plot = False, record_data = True, observe_direction = True,
            track_birds = False, sim_length = 5000, learning_alg = 'Q', comment = 'vary_gamma',
            alpha = 0.9, gamma = gamma
        )
