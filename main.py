from field import Field

if __name__ == '__main__':
    Field(
        100, periodic = True, plot = False, record_data = True, observe_direction = True,
        sim_length = 5000, learning_alg = 'Q', alpha = 0.9, gamma = 0.9
    )
