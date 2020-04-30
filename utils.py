from os import path
from datetime import datetime
import regex

re_tag = regex.compile(r'^[0-9]+-[0-9]+|^desired')

def gen_rt():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def get_rt(fpath):
    fname = path.split(fpath)[1]
    return re_tag.search(fname).group()
