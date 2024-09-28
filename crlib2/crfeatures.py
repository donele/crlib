import os
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from math import log2

from crlib2.crdataio import *

def future_returns(prc, ri):
    r = None
    if prc is not None and len(prc) > ri:
        r = (prc.shift(-2**ri) / prc - 1).fillna(0).replace([np.inf, -np.inf], 0)
    return r

def past_returns(prc, ri):
    r = None
    if prc is not None and len(prc) > ri:
        r = (prc / prc.shift(2**ri) - 1).fillna(0).replace([np.inf, -np.inf], 0)
    return r

def get_returns(df, sample_timex, max_timex, varname, returns_func):
    '''
    Calculates the future returns.

    ri: Relative index.
    ai: Absolute index.
    '''
    serdict = {}
    for ri in range(max_timex - 1):
        ser = returns_func(df.mid, ri)
        ai = ri + sample_timex
        ser.name = f'{varname}_{ai}'
        serdict[ser.name] = ser

    for ri in range(max_timex - 2):
        ai = ri + sample_timex
        for rj in range(ri + 1, max_timex - 1):
            if varname == 'ret' and rj > ri + 1:
                continue
            aj = rj + sample_timex
            name1 = f'{varname}_{ai}'
            name2 = f'{varname}_{aj}'
            ser = serdict[name2] - serdict[name1]
            ser.name = f'diff_{varname}_{ai}_{aj}'
            serdict[ser.name] = ser

    serlist = list(serdict.values())
    return serlist

def features_future_returns(df, sample_timex, max_timex=10):
    return get_returns(df, sample_timex, max_timex, 'tar', returns_func=future_returns)

def features_past_returns(df, sample_timex, max_timex=10):
    return get_returns(df, sample_timex, max_timex, 'ret', returns_func=past_returns)

def get_features(dft, dfb, dfm, sample_timex, mid_col, index_col, max_timex=None,
                 max_tsince_trade=5, verbose=False):
    '''
    Calculates the features.

    sample_timex:
		0:     1 us
        1:     2 us
        2:     4 us
        3:     8 us
        4:    16 us
        5:    32 us
        6:    64 us
        7:   128 us
        8:   256 us
        9:   512 us
       10:   1.0 ms
       11:   2.0 ms
       12:   4.1 ms
       13:   8.2 ms
       14:  16.4 ms
       15:  32.8 ms
       16:  65.5 ms
       17: 131.1 ms
       18: 262.1 ms
       19: 524.3 ms
       20:   1.0 s
       21:   2.1 s
       22:   4.2 s
       23:   8.4 s
       24:  16.8 s
       25:  33.6 s
       26:   1.1 m
       27:   2.2 m
       28:   4.5 m
       29:   8.9 m
       30:  17.9 m
       31:  35.8 m
       32:  71.6 m

    Returns:
        A dataframe.
    '''
    sample_interval = int(2**int(sample_timex))

    tgrp = pd.to_datetime(dft[index_col] // sample_interval * sample_interval, unit='us')
    dftagg = dft.groupby(tgrp).agg(
        price=('price', 'last'),
        avg_price=('price', 'mean'),
        min_price=('min_px', 'min'),
        max_price=('max_px', 'max'),
        sum_avg_qty=('abs_qty', 'sum'),
        sum_net_qty=('net_qty', 'sum'),
        last_trade=('price', lambda x: x.index[-1]),
    )
    if verbose:
        print(dftagg)

    dfb['qimb'] = ((dfb.askqty - dfb.bidqty) / (dfb.askqty + dfb.bidqty)).fillna(0).replace([np.inf, -np.inf], 0)

    bgrp = pd.to_datetime(dfb[index_col] // sample_interval * sample_interval, unit='us')
    dfbagg = dfb.groupby(bgrp).agg(
        qimb=('qimb', 'last'),
        adj_askpx=('adj_askpx', 'last'),
        adj_bidpx=('adj_bidpx', 'last'),
        max_askqty=('askqty', 'max'),
        max_bidqty=('bidqty', 'max'),
    )

    mgrp = pd.to_datetime(dfm[index_col] // sample_interval * sample_interval, unit='us')
    dfmagg = dfm.groupby(mgrp).agg(
        adj_width=('adj_width', 'last'),
        adj_mid_px=('adj_mid_px', 'last'),
    )

    # Merge trade and bbo

    allindx = pd.date_range(dfbagg.index[0], dfbagg.index[-1], freq=timedelta(microseconds=sample_interval))
    df = pd.concat([dfbagg.reindex(allindx), dftagg.reindex(allindx), dfmagg.reindex(allindx)], axis=1)

    # mid price, to be used for some feature calculation.
    # df['mid'] = df['adj_askpx'] - df['adj_bidpx'] # this can be negative!
    df['mid'] = df[mid_col] # Synthetic price from a model.

    # ffill
    ffill_cols = ['mid', 'adj_askpx', 'adj_bidpx', 'adj_width', 'last_trade']
    df[ffill_cols] = df[ffill_cols].ffill()

    # tsince_trade
    df['tsince_trade'] = (df.index.to_series().shift(-1) - df.last_trade.fillna(datetime.datetime(1970,1,1))).map(lambda x: x.total_seconds())
    df['valid'] = (df.last_trade.notna()) & (df.tsince_trade <= max_tsince_trade)

    serlst = []


    if max_timex is None:
        timex_1hr = int(log2(60*60*1e6)) # ~1 hour.
        max_timex = timex_1hr - sample_timex

    # BBO related features

    ## Future returns
    serlst.extend(features_future_returns(df, sample_timex, max_timex))

    ## Past returns
    serlst.extend(features_past_returns(df, sample_timex, max_timex))

    ## Median qimb

    for ri in range(max_timex):
        w = 2**ri
        ser = df.qimb.rolling(window=w, min_periods=1).median().fillna(0).replace([np.inf, -np.inf], 0)
        ai = ri + sample_timex
        ser.name = f'medqimb_{ai}'
        serlst.append(ser)

    ## qimax

    for ri in range(max_timex):
        w = 2**ri
        ai = ri + sample_timex
        aname = f'max_askqty_{ai}'
        bname = f'max_bidqty_{ai}'
        aser = df.max_askqty.rolling(window=w, min_periods=1).max()
        bser = df.max_bidqty.rolling(window=w, min_periods=1).max()
        aser.name = aname
        bser.name = bname
        qiser = ((aser - bser) / (aser + bser)).fillna(0).replace([np.inf, -np.inf], 0)
        qiser.name = f'qimax_{ai}'
        serlst.append(qiser)

    ## Volatility

    volatlist = []
    rser = df.price / df.price.shift(1) - 1
    for ri in range(2, max_timex):
        w = 2**ri
        ser = rser.rolling(window=w, min_periods=1).std()
        ai = ri + sample_timex
        ser.name = f'volat_{ai}'
        volatlist.append(ser)
    dfvolat = pd.DataFrame(volatlist).T

    for ri in range(2, max_timex):
        ai = ri + sample_timex
        for rj in range(max(4, ri + 1), max_timex):
            aj = rj + sample_timex
            name1 = f'volat_{ai}'
            name2 = f'volat_{aj}'
            ser = ((dfvolat[name1] - dfvolat[name2]) / dfvolat[name2].abs()).fillna(0).replace([np.inf, -np.inf], 0)
            ser.name = f'diff_volat_{ai}_{aj}'
            serlst.append(ser)

    # Trade related feature

    ## Hilo

    hilolst = []
    for ri in range(max_timex):
        w = 2**ri
        hiser = df.max_price.rolling(window=w, min_periods=1).max()
        loser = df.min_price.rolling(window=w, min_periods=1).min()
        ai = ri + sample_timex
        hiser.name = f'max_price_{ai}'
        loser.name = f'min_price_{ai}'
        hilolst.append(hiser)
        hilolst.append(loser)
    dfhl = pd.DataFrame(hilolst).T
    for ri in range(max_timex):
        ai = ri + sample_timex
        hiname = f'max_price_{ai}'
        loname = f'min_price_{ai}'
        ser = ((df.price - (.5*dfhl[hiname] + .5*dfhl[loname])) / (.5*dfhl[hiname] - .5*dfhl[loname])).fillna(0).replace([np.inf, -np.inf], 0)
        ser.name = f'hilo_{ai}'
        serlst.append(ser)


    # Volume related feature

    avglst = []
    netlst = []

    for ri in range(max_timex):
        w = 2**ri
        aser = df.sum_avg_qty.rolling(window=w, min_periods=1).sum()
        nser = df.sum_net_qty.rolling(window=w, min_periods=1).sum()
        ai = ri + sample_timex
        aser.name = f'sum_avg_qty_{ai}'
        nser.name = f'sum_net_qty_{ai}'
        avglst.append(aser)
        netlst.append(nser)
    dfavg = pd.DataFrame(avglst).T
    dfnet = pd.DataFrame(netlst).T

    for ri in range(5):
        ai = ri + sample_timex
        for rj in range(ri + 1, max_timex):
            aj = rj + sample_timex
            name1 = f'sum_avg_qty_{ai}'
            name2 = f'sum_avg_qty_{aj}'
            ser = ((dfavg[name1] - dfavg[name2]) / dfavg[name2].abs()).fillna(0).replace([np.inf, -np.inf], 0)
            ser.name = f'diff_sum_avg_qty_{ai}_{aj}'
            serlst.append(ser)

    for ri in range(5):
        ai = ri + sample_timex
        for rj in range(ri + 1, max_timex):
            aj = rj + sample_timex
            name1 = f'sum_net_qty_{ai}'
            name2 = f'sum_net_qty_{aj}'
            name3 = f'sum_avg_qty_{aj}'
            ser = ((dfnet[name1] - dfnet[name2]) / dfavg[name3].abs()).fillna(0).replace([np.inf, -np.inf], 0)
            ser.name = f'diff_sum_net_qty_{ai}_{aj}'
            serlst.append(ser)

    # Concat all

    df = pd.concat([df] + serlst, axis=1)
    return df

