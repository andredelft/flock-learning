from field import Field

if __name__ == '__main__':
    Field(
        100, periodic = True, plot = True, record_data = False, observe_direction = True,
        track_birds = False, sim_length = 15000, learning_alg = 'Ried'
    )
