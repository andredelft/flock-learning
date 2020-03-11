from field import Field

if __name__ == '__main__':
    Field(
        500, periodic = True, plot = True, record_data = True, observe_direction = True,
        track_birds = True, sim_length = 5000, learning_alg = 'Q', comment = 'vary_alpha',
        alpha = 0.9, gamma = 0.9
    )
