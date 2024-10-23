import os
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from math import log2

from .crdataio import *

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

def get_returns(df, sample_timex, min_timex, max_timex, varname, returns_func, col='mid', alldiff=False):
    '''
    Calculates the future returns.

    ri: Relative index.
    ai: Absolute index.
    '''
    serdict = {}
    for ri in range(min_timex, max_timex - 1):
        ser = returns_func(df[col], ri)
        ai = ri + sample_timex
        ser.name = f'{varname}_{ai}'
        serdict[ser.name] = ser

    for ri in range(min_timex, max_timex - 2):
        ai = ri + sample_timex
        for rj in range(ri + 1, max_timex - 1):
            if alldiff or rj == ri + 1:
                aj = rj + sample_timex
                name1 = f'{varname}_{ai}'
                name2 = f'{varname}_{aj}'
                ser = serdict[name2] - serdict[name1]
                ser.name = f'diff_{varname}_{ai}_{aj}'
                serdict[ser.name] = ser

    serlist = list(serdict.values())
    return serlist

def features_future_returns(df, sample_timex, min_timex=0, max_timex=10):
    return get_returns(df, sample_timex, min_timex, max_timex, 'tar', returns_func=future_returns, alldiff=True)

def features_past_returns(df, sample_timex, min_timex=0, max_timex=10):
    return get_returns(df, sample_timex, min_timex, max_timex, 'ret', returns_func=past_returns, alldiff=False)

def features_qimb(df, sample_timex, min_timex=0, max_timex=10):
    return get_returns(df, sample_timex, min_timex, max_timex, 'qimbret', returns_func=past_returns, col='qimb', alldiff=False)

def features_sprdrat(df, sample_timex, min_timex=0, max_timex=10):
    return get_returns(df, sample_timex, min_timex, max_timex, 'sprdratret', returns_func=past_returns, col='sprdrat', alldiff=False)

def get_timegroup(tser, sample_interval):
    '''
    Calculate the timestamps at the end of the time bars.
    '''
    return pd.to_datetime((tser // sample_interval + 1) * sample_interval, unit='us')

def make_features(dft, dfb, dfm, sample_timex, mid_col, index_col, min_timex=None, max_timex=None,
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

    dft['tlat'] = dft.t0 - dft.max_xts

    tgrp = get_timegroup(dft[index_col], sample_interval)
    dftagg = dft.groupby(tgrp).agg(
        tlat = ('tlat', 'last'),
        price = ('price', 'last'),
        avg_price = ('price', 'mean'),
        min_price = ('min_px', 'min'),
        max_price = ('max_px', 'max'),
        sum_avg_qty = ('abs_qty', 'sum'),
        sum_net_qty = ('net_qty', 'sum'),
        last_trade = ('price', lambda x: x.index[-1]),
    )
    if verbose:
        print(dftagg)

    dfb['qimb'] = ((dfb.askqty - dfb.bidqty) / (dfb.askqty + dfb.bidqty)).fillna(0).replace([np.inf, -np.inf], 0)
    dfb['width'] = (dfb.askpx - dfb.bidpx)
    dfb['sprdrat'] = (dfb.askpx - dfb.bidpx) / (.5*dfb.askpx + .5*dfb.bidpx)
    dfb['blat'] = dfb.t0 - dfb.xts

    bgrp = get_timegroup(dfb[index_col], sample_interval)
    dfbagg = dfb.groupby(bgrp).agg(
        blat = ('blat', 'last'),
        qimb = ('qimb', 'last'),
        askpx = ('askpx', 'last'),
        bidpx = ('bidpx', 'last'),
        width = ('width', 'last'),
        sprdrat = ('sprdrat', 'last'),
        adj_askpx = ('adj_askpx', 'last'),
        adj_bidpx = ('adj_bidpx', 'last'),
        max_askqty = ('askqty', 'max'),
        max_bidqty = ('bidqty', 'max'),
    )

    mgrp = get_timegroup(dfm[index_col], sample_interval)
    dfmagg = dfm.groupby(mgrp).agg(
        adj_width = ('adj_width', 'last'),
        adj_mid_px = ('adj_mid_px', 'last'),
    )

    # Merge trade and bbo

    allindx = pd.date_range(dfbagg.index[0], dfbagg.index[-1], freq=timedelta(microseconds=sample_interval))
    df = pd.concat([dfbagg.reindex(allindx), dftagg.reindex(allindx), dfmagg.reindex(allindx)], axis=1)

    # mid price, to be used for some feature calculation.
    # df['mid'] = df['adj_askpx'] - df['adj_bidpx'] # this can be negative!
    if mid_col == 'bidask':
        df['mid'] = .5*df['askpx'] + .5*df['bidpx']
    else:
        df['mid'] = df[mid_col]

    # ffill
    ffill_cols = ['mid', 'width', 'adj_askpx', 'adj_bidpx', 'adj_width', 'last_trade']
    df[ffill_cols] = df[ffill_cols].ffill()

    # tsince_trade
    df['tsince_trade'] = (df.index.to_series().shift(-1) - df.last_trade.fillna(datetime.datetime(1970,1,1))).map(lambda x: x.total_seconds())
    df['valid'] = (df.last_trade.notna()) & (df.tsince_trade <= max_tsince_trade)

    serlst = []

    if min_timex is None:
        min_timex = 0

    if max_timex is None:
        timex_1hr = int(log2(60*60*1e6)) # ~1 hour.
        max_timex = timex_1hr - sample_timex

    # BBO related features

    ## Future returns
    serlst.extend(features_future_returns(df, sample_timex, min_timex, max_timex))

    ## Past returns
    serlst.extend(features_past_returns(df, sample_timex, min_timex, max_timex))

    if False: # Can be turn on as needed.
        serlst.extend(features_qimb(df, sample_timex, min_timex, max_timex))
        serlst.extend(features_sprdrat(df, sample_timex, min_timex, max_timex))

    ## Median qimb

    for ri in range(min_timex, max_timex):
        w = 2**ri
        ser = df.qimb.rolling(window=w, min_periods=1).median().fillna(0).replace([np.inf, -np.inf], 0)
        ai = ri + sample_timex
        ser.name = f'medqimb_{ai}'
        serlst.append(ser)

    ## qimax

    for ri in range(min_timex, max_timex):
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
    for ri in range(2 + min_timex, max_timex):
        w = 2**ri
        ser = rser.rolling(window=w, min_periods=1).std()
        ai = ri + sample_timex
        ser.name = f'volat_{ai}'
        volatlist.append(ser)
    dfvolat = pd.DataFrame(volatlist).T

    for ri in range(2 + min_timex, max_timex):
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
    for ri in range(min_timex, max_timex):
        w = 2**ri
        hiser = df.max_price.rolling(window=w, min_periods=1).max()
        loser = df.min_price.rolling(window=w, min_periods=1).min()
        ai = ri + sample_timex
        hiser.name = f'max_price_{ai}'
        loser.name = f'min_price_{ai}'
        hilolst.append(hiser)
        hilolst.append(loser)
    dfhl = pd.DataFrame(hilolst).T
    for ri in range(min_timex, max_timex):
        ai = ri + sample_timex
        hiname = f'max_price_{ai}'
        loname = f'min_price_{ai}'
        ser = ((df.price - (.5*dfhl[hiname] + .5*dfhl[loname])) / (.5*dfhl[hiname] - .5*dfhl[loname])).fillna(0).replace([np.inf, -np.inf], 0)
        ser.name = f'hilo_{ai}'
        serlst.append(ser)


    # Volume related feature

    avglst = []
    netlst = []

    for ri in range(min_timex, max_timex):
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

    for ri in range(min_timex, 5):
        ai = ri + sample_timex
        for rj in range(ri + 1, max_timex):
            aj = rj + sample_timex
            name1 = f'sum_avg_qty_{ai}'
            name2 = f'sum_avg_qty_{aj}'
            ser = ((dfavg[name1] - dfavg[name2]) / dfavg[name2].abs()).fillna(0).replace([np.inf, -np.inf], 0)
            ser.name = f'diff_sum_avg_qty_{ai}_{aj}'
            serlst.append(ser)

    for ri in range(min_timex, 5):
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

def get_reg_features_3h(dt1, par, min_timex=None, max_timex=None):
    '''
    Calculate the features in a three hour period, then truncate to oen hour period.
    '''
    index_col = par['index_col'] if 'index_col' in par else None
    mid_col = par['mid_col'] if 'mid_col' in par else None
    sample_timex = par['sample_timex'] if 'sample_timex' in par else par['grid_timex'] if 'grid_timex' in par else None

    dft = read_3h('trade', dt1, par)
    dfb = read_3h('bbo', dt1, par)
    dfm = read_3h('midpx', dt1, par)
    if dft is None or dfb is None or dfm is None:
        return None
    df = make_features(dft, dfb, dfm, sample_timex, mid_col, index_col, min_timex, max_timex)
    return df

def get_tevt_features_3h(dt, par, min_timex=None, max_timex=None, min_row=100):
    '''
    Calculate the features in a three hour period, then truncate to oen hour period.
    '''
    if min_timex is None:
        min_timex = par['min_feature_timex'] - par['grid_timex']
    dff = get_reg_features_3h(dt, par, min_timex, max_timex)

    dft = read_3h('trade', dt, par)
    if dft is None or len(dft) < min_row:
        return None

    # Time weighted moving average
    min_twma_timex = 10
    max_twma_timex = 30
    timex_list = np.array(range(min_twma_timex, max_twma_timex))
    dfma = pd.DataFrame(columns=[f'twma_{x}' for x in timex_list], index=dft.index)
    dfma.iloc[0] = dft.price[0]
    timeconsts = np.array([2.0**x for x in timex_list])
    timediff = dft.t0.diff().fillna(0)
    prc = dft.price

    # Price MA
    for i in range(1, len(dft)):
        dfma.iloc[i] = dfma.iloc[i-1] + np.minimum(timeconsts, timediff.iloc[i]) / timeconsts * (prc[i] - dfma.iloc[i-1])

    # twret
    twret = pd.DataFrame((dft.price.to_numpy().reshape(-1,1) / dfma.to_numpy() - 1),
                    index=dft.index, columns=[f'twret_{x}' for x in timex_list]).fillna(0)
    del dfma
    dft = pd.concat([dft, twret], axis=1)

    # Determine samples and drop non-sample trades
    dft['sample'] = False
    dft = dft.loc[~dft.index.duplicated(keep='first')]
    last_sample = 0
    for idx, row in dft.iterrows():
        if row.t0 > last_sample + 5000:
            dft.loc[idx, 'sample'] = True
            last_sample = row.t0
    dft = dft[dft['sample']]

    # Merge grid and trade, ffill, then select samples only
    dfmer = pd.merge(dff, dft, how='outer', left_index=True, right_index=True, suffixes=('', '_trd'))
    dfmer[dff.columns] = dfmer[dff.columns].ffill()
    dfmer = dfmer[(dfmer['valid'].shift() == True)&(dfmer['sample'] == True)]

    # Recalculate ret
    retnames = [x for x in dfmer.columns if x.startswith('ret_')]
    retidxlist = [int(x[-2:]) for x in retnames if len(x) == 6 and x[3] == '_']
    for name in retnames:
        dfmer[name] *= dfmer.price_trd / dfmer.price
        dfmer[name] += dfmer.price_trd / dfmer.price - 1

    # Recalculate tar
    tarnames = [x for x in dfmer.columns if x.startswith('tar_')]
    for name in tarnames:
        dfmer[name] *= dfmer.price / dfmer.price_trd
        dfmer[name] += dfmer.price / dfmer.price_trd - 1

    return dfmer

def get_features_3h(dt, par, min_timex=None, max_timex=None):
    df = None
    if 'sample_type' in par and par['sample_type'] == 'tevt':
        df = get_tevt_features_3h(dt, par)
    else:
        df = get_reg_features_3h(dt, par)
    return df
