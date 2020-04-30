from os import path

def get_rt(fpath):
    fname = path.split(fpath)[1]
    return '-'.join(fname.split('-')[:2])
