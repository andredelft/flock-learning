from field import Field

if __name__ == '__main__':
    for r in [1,2,5,10]:
        Field(
            100, periodic = True, plot = True, record_data = True, observe_direction = True,
            sim_length = 12500, learning_alg = 'Ried', comment = 'vary:reward_signal',
            reward_signal = r#, alpha = 0.9, gamma = 0.9
        )
