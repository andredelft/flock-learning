from field import Field
from matplotlib import pyplot as plt

def avg(data, cap = 20):
    return [sum(data[i:i + cap])/cap for i in range(len(data) - cap)]

if __name__ == '__main__':
    Field(100, periodic = True, plot = False, record = True)
