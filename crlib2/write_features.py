import argparse
import datetime
import multiprocessing as mp

from crlib2 import *
from crlib2.crfeatures import *

def write_features():
    parser = argparse.ArgumentParser(prog='fit1d')
    parser.add_argument('-s', '--start-date', type=datetime.datetime.fromisoformat, required=True,
            metavar='yyyy-mm-dd.hh')
    parser.add_argument('-e', '--end-date', type=datetime.datetime.fromisoformat, required=True,
            metavar='yyyy-mm-dd.hh')
    parser.add_argument('-n', '--num-threads', type=int, required=False, default=0, help='Number of parallel processes')

    a = parser.parse_args()

    par = read_params('params.yaml')
    print(par)
    
    st = a.start_date.replace(minute=0, second=0, microsecond=0)
    et = a.end_date.replace(minute=0, second=0, microsecond=0)
    dr = pd.date_range(st, et, freq='h')
    if a.num_threads <= 1:
        for dt in dr:
            write_feature(dt, par)
    elif a.num_threads > 1:
        pool = mp.Pool(processes=a.num_threads)
        results = [pool.apply_async(write_feature, args=(dt, par)) for dt in dr]
        pool.close()
        pool.join()
