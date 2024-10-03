import os
import numpy as np
import pandas as pd
from datetime import timedelta

crdata_map = {
    'default': {0: '/mnt/bigdata1/crdata',
            20240601: '/mnt/bigdata2/crdata',
             },
}

def parse_dt(dt):
    '''
    Returns yyyymmdd and hh.
    '''
    yyyymmdd = dt.year * 10000 + dt.month * 100 + dt.day
    hh = dt.hour
    return yyyymmdd, hh

def get_data_dir(dt, locale):
    yyyy = dt.year
    mmdd = dt.month * 100 + dt.day
    yyyymmdd = yyyy * 10000 + mmdd

    switchmap = crdata_map[locale] if locale in crdata_map else crdata_map['default']
    switchdate = 0
    for k, v in switchmap.items():
        if k > switchdate and k <= yyyymmdd:
            switchdate = k
    disk = switchmap[switchdate]

    data_dir = f'{disk}/{locale}.{yyyy}/{mmdd:04}'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    return data_dir

def read_data(datatype, dt, locale, index_col, columns=None, product=None):
    '''
    Reads the tick data of the specified datatype for the time dt.
    '''
    data_dir = get_data_dir(dt, locale)
    yyyymmdd, hh = parse_dt(dt)
    path = f'{data_dir}/{datatype}.{locale}.{yyyymmdd}-{hh:02d}.parquet.snappy'
    if not os.path.exists(path):
        print(path, ' not found')
        return None

    product_cols = ['exchange', 'symbol']
    readcols = None if columns is None else product_cols + columns
    df = pd.read_parquet(path, columns=readcols)

    if index_col is None:
        pass
    elif product is None:
        df['date'] = pd.to_datetime(df[index_col], unit='us')
        df = df.set_index(product_cols)
    else:
        df = df[(df.exchange==product[0]) & (df.symbol==product[1])]
        df['date'] = pd.to_datetime(df[index_col], unit='us')
        df = df.set_index('date').sort_index()
    df = df.drop(columns=[x for x in df.columns if columns is not None and x not in columns])

    return df

def read_trade(dt, locale, index_col,
               columns=['min_px', 'max_px', 'price', 'abs_qty', 'net_qty', 't0', 'ts'], product=None):
    '''
    Reads the trade data for the time dt.
    '''
    df = read_data('trade', dt, locale, index_col, columns, product)
    return df

def read_bbo(dt, locale, index_col,
             columns=['askpx', 'askqty', 'bidpx', 'bidqty', 'adj_askpx', 'adj_bidpx', 'imbalance', 't0', 'ts'], product=None):
    '''
    Reads the bbo data for the time dt.
    '''
    df = read_data('bbo', dt, locale, index_col, columns, product)
    return df

def read_midpx(dt, locale, index_col,
               columns=['mid_px', 'adj_width', 'adj_mid_px', 't0', 'ts'], product=None):
    '''
    Reads the midpx data for the time dt.
    '''
    df = read_data('midpx', dt, locale, index_col, columns, product)
    return df

def read_func(datatype):
    func = None
    if datatype == 'trade':
        func = read_trade
    elif datatype == 'bbo':
        func = read_bbo
    elif datatype == 'midpx':
        func = read_midpx
    return func

def read_range(datatype, st, et, par=None):
    '''
    Reads the tick data of the specified type within the specified time range st and et.
    '''
    locale = par['locale'] if 'locale' in par else None
    index_col = par['index_col'] if 'index_col' in par else None
    product = par['product'] if 'product' in par else None

    func = read_func(datatype)
    dt_range = pd.date_range(st, et, freq='h')
    datalist = []
    for dt in dt_range:
        df0 = func(dt, locale, index_col, product=product)
        datalist.append(df0)

    if np.all([x is None for x in datalist]):
        return None
    df = pd.concat(datalist)
    return df

def read_3h(datatype, dt1, par=None):
    '''
    Reads the data in three consecutive periods from before dt1 to after dt1.
    '''
    locale = par['locale'] if 'locale' in par else None
    index_col = par['index_col'] if 'index_col' in par else None
    product = par['product'] if 'product' in par else None

    func = read_func(datatype)

    dt0 = dt1 - timedelta(hours=1)
    dt2 = dt1 + timedelta(hours=1)

    datalist = []
    datalist.append(func(dt0, locale, index_col, product=product))
    datalist.append(func(dt1, locale, index_col, product=product))
    datalist.append(func(dt2, locale, index_col, product=product))

    if np.any([x is None for x in datalist]):
        return None
    df = pd.concat(datalist)
    return df
