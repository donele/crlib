import pickle
import pandas as pd

from crlib2 import *

def test_pnl():
    dfo = pd.read_pickle('dfo.pkl')
    pos = pd.read_pickle('pos.pkl')
    _, testpos, _ = get_pnl(dfo, 'tar_19', 0)
    assert np.all(pos == testpos)
