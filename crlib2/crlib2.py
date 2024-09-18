import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import glob
import datetime
import seaborn as sns
import yaml
from datetime import timedelta
from math import *

import lightgbm as lgb
from sklearn.linear_model import LinearRegression

## Data Hangling

crdata_map = {
	'japan': {0: '/media/jdlee/bigdata1/crdata',
             },
    'default': {0: '/media/jdlee/bigdata1/crdata',
             },
}
crdata_map_bk = {
	'japan': {0: '/media/jdlee/datadisk0/crdata',
              20240201: '/media/jdlee/datadisk1/crdata',
             },
    'default': {0: '/media/jdlee/datadisk0/crdata',
                20240101: '/media/jdlee/datadisk2/crdata',
                20240501: '/media/jdlee/datadisk3/crdata',
             },
}

dfuniv = None
upath = 'universe.csv'
if os.path.exists(upath):
    dfuniv = pd.read_csv(upath, index_col=[0, 1])

def get_data_dir(dt, locale):
    yyyy = dt.year
    mmdd = dt.month * 100 + dt.day
    yyyymmdd = yyyy * 10000 + mmdd

    switchmap = crdata_map[locale] if locale in crdata_map else crdata_map['default']
    switchdate = 0
    for k, v in switchmap.items():
        if k >= switchdate and k < yyyymmdd:
            switchdate = k
    disk = switchmap[switchdate]

    data_dir = f'{disk}/{locale}.{yyyy}/{mmdd:04}'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    return data_dir

def read_params(path):
    '''
    Reads the project parameters from yaml file.

    Returns:
        A dictionary.
    '''
    par = None
    if os.path.exists(path):
        ext = path.split('.')[-1]
        if ext == 'yaml':
            with open(path) as f:
                par = yaml.safe_load(f)
    return par

def get_universe(idx=0):
    '''
    Selects liquid produc.
    '''
    product = dfuniv.index[idx]
    return product

def get_idate(dt):
    '''
    Returns the date in yyyymmdd format.
    '''
    idate = int(dt.strftime('%Y%m%d'))
    return idate

def parse_dt(dt):
    '''
    Returns yyyymmdd and hh.
    '''
    yyyymmdd = dt.year * 10000 + dt.month * 100 + dt.day
    hh = dt.hour
    return yyyymmdd, hh

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
               columns=['price', 'abs_qty', 'net_qty', 't0', 'ts'], product=None):
    '''
    Reads the trade data for the time dt.
    '''
    df = read_data('trade', dt, locale, index_col, columns, product)
    return df

def read_bbo(dt, locale, index_col,
             columns=['askpx', 'askqty', 'bidpx', 'bidqty', 'adj_askpx', 'adj_bidpx', 't0', 'ts'], product=None):
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

def read3(datatype, dt1, par=None):
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

## Features

def get_timeindex(interval):
    timeindex = 0
    while sample_interval >= 10:
        sample_interval //= 10
        timeindex += 3

    if sample_interval == 1:
        pass
    elif sample_interval == 2:
        timeindex += 1
    elif sample_interval == 5:
        timeindex += 2
    else:
        timeindex = None
    return timeindex

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
        min_price=('price', 'min'),
        max_price=('price', 'max'),
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

    allindx = pd.date_range(dfbagg.index[0], dfbagg.index[-1], freq=datetime.timedelta(microseconds=sample_interval))
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

    hllst = []
    for ri in range(max_timex):
        w = 2**ri
        hiser = df.max_price.rolling(window=w, min_periods=1).max()
        loser = df.min_price.rolling(window=w, min_periods=1).min()
        ai = ri + sample_timex
        hiser.name = f'max_price_{ai}'
        loser.name = f'min_price_{ai}'
        hllst.append(hiser)
        hllst.append(loser)
    dfhl = pd.DataFrame(hllst).T
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

def get_features3(dt1, par, max_timex=None):
    '''
    Reads the data in three consecutive periods
    '''
    index_col = par['index_col'] if 'index_col' in par else None
    mid_col = par['mid_col'] if 'mid_col' in par else None
    sample_timex = par['sample_timex'] if 'sample_timex' in par else None

    dt2 = dt1 + timedelta(hours=1)

    dft = read3('trade', dt1, par)
    dfb = read3('bbo', dt1, par)
    dfm = read3('midpx', dt1, par)
    if dft is None or dfb is None or dfm is None:
        return None
    df = get_features(dft, dfb, dfm, sample_timex, mid_col, index_col, max_timex)
    df = df.loc[dt1:(dt2-timedelta(microseconds=1))]
    return df

def get_feature_dir(par, basedir='/home/jdlee'):
    exch = par['product'][0]
    sym = par['product'][1]
    pname = par['proj_name']
    feature_dir = f'{basedir}/crfeature/{exch}.{sym}.{pname}'
    return feature_dir

def get_fit_dir(par, basedir='/home/jdlee'):
    exch = par['product'][0]
    sym = par['product'][1]
    pname = par['proj_name']
    fit_dir = f'{basedir}/crfit/{exch}.{sym}.{pname}'
    return fit_dir

def write_feature(dt, par):
    df0 = get_features3(dt, par)
    if df0 is not None and df0.shape[0] > 0:
        feature_dir = get_feature_dir(par)
        if not os.path.isdir(feature_dir):
            try:
                os.makedirs(feature_dir)
            except:
                return None
        yyyymmdd, hh = parse_dt(dt)
        df0.to_parquet(f'{feature_dir}/{yyyymmdd}.{hh:02d}.parquet')
    return df0.shape if df0 is not None else None

def read_features(dt1, dt2, feature_dir, features=None):
    '''
    Reads the features data in the time range between dt1 and dt2.

    Returns:
        A dataframe.
    '''
    dt1 = dt1.replace(minute=0, second=0, microsecond=0)
    dt2 = dt2.replace(minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
    dr = pd.date_range(dt1, dt2, freq='h')
    dflist = []
    for dt in dr:
        yyyymmdd, hh = parse_dt(dt)
        path = f'{feature_dir}/{yyyymmdd}.{hh:02d}.parquet'
        if not os.path.exists(path):
            print(path, ' not found.')
            continue
        df = pd.read_parquet(path, columns=features)
        if df is not None and df.shape[0] > 0:
            dflist.append(df)
    if len(dflist) > 0:
        df = pd.concat(dflist)
        return df
    return None

def plot_test_features(dff):
    tarcols = [x for x in dff.columns if x.startswith('tar')]
    ncol = len(tarcols)
    nx = int(ncol**.5) + 1
    ny = nx + 1

    dff[tarcols].hist(figsize=(16,2*ny), bins=80)
    dff[tarcols].hist(figsize=(16,2*ny), bins=80, log=True)

    feature_groups = ['ret', 'medqimb', 'qimax', 'hilo', 'diff_sum_avg_qty', 'diff_sum_net_qty']
    for fg in feature_groups:
        cols = [x for x in dff.columns if x.startswith(fg+'_')]
        dff[cols].hist(figsize=(16,12), bins=40)

        plt.figure(figsize=(16,4))

        plt.subplot(121)
        dfcorr = dff[cols+tarcols].corr().iloc[:len(cols), len(cols):]
        sns.heatmap(dfcorr, cmap='RdYlGn_r', linewidths=0.5, annot=False)

        plt.subplot(122)
        nc = 5
        tar = tarcols[0]
        for col in cols:
            qc = pd.qcut(dff[col], nc, duplicates='drop')
            plt.plot(dff.groupby(qc)[tar].mean().tolist(), label=col)
        plt.grid()
        if(len(col) < 10):
            plt.legend()

## Linear Regression

class BasicLinearModel:
    def __init__(self, feature_names, coefs):
        self.feature_names_ = feature_names
        self.coefs_ = coefs
    def predict(self, X):
        X_ = X.copy()
        X_['const'] = 1
        pred = X_ @ self.coefs_
        sys.stdout.flush()
        return pred

def linreg(target_name, dft, fit_features, verbose=False):
    '''
    Performs a OLS minimization.

    Returns:
        Linear regression coefficients.
    '''
    X = dft.loc[dft.valid, fit_features] # this will be copied inside LinearRegression
    y = dft.loc[dft.valid, target_name]

    model = LinearRegression()
    model.fit(X, y)
    return model

def linreg_select_features(target_name, dft, dfv, metric='r2', feature_groups=None,
        verbose=False, debug_nfeature=None, min_data_cnt=10):
    '''
    Select feaures recursively using validation sample.

    Returns:
        Series of selected features and the regression coefficients.
    '''
    selected_features = []
    best_model = None
    if (dft is None or dfv is None or dft.loc[dft.valid].shape[0] < min_data_cnt
        or dfv.loc[dfv.valid].shape[0] < min_data_cnt):
        return best_model

    if feature_groups is None:
        feature_groups=['ret', 'medqimb', 'qimax', 'hilo']

    allfeatures = [x for x in dft.columns if np.any([x.startswith(g+'_') for g in feature_groups])]
    if not debug_nfeature is None:
        allfeatures = allfeatures[:min(len(allfeatures), debug_nfeature)]

    print(f'Total {len(allfeatures)} features: {allfeatures}')

    remaining_features = allfeatures.copy()

    thres = 0.01
    best_metric = 0
    while True:
        new_metric = 0
        new_model = None
        new_feature = None
        for feature in remaining_features:
            # Fit
            fit_features = selected_features + [feature]
            model = linreg(target_name, dft, fit_features, verbose)
            if model is None:
                continue
            else:
                Xv = dfv.loc[dfv.valid, fit_features]
                yv = dfv.loc[dfv.valid, target_name]
                yhatv = pd.Series(model.predict(Xv), index=yv.index)

                # Validation
                r2 = 1 - np.var(yv - yhatv) / np.var(yv)
                rmse = -1e4*(((yv - yhatv)**2).mean()**.5)
                ev = 1e4*(.5*yv[yhatv > yhatv.quantile(0.99)].mean() - .5*yv[yhatv < yhatv.quantile(0.01)].mean())

                # Check if new best is found
                metric_val = r2 if metric == 'r2' else rmse if metric == 'rmse' else ev if metric == 'ev' else 0
                if metric_val > new_metric and (metric_val - best_metric) > abs(best_metric) * thres:
                    new_metric = metric_val
                    new_model = model
                    new_feature = feature

                if verbose:
                    print(f'{fit_features} r2={r2:.4f} rmse={rmse:.4f}, ev={ev:.4f}')

        if new_feature is None:
            break
        else:
            print(f'Selected {new_feature} {metric}={new_metric:.4f}')
            sys.stdout.flush()
            best_metric = new_metric
            best_model = new_model
            selected_features.append(new_feature)
            remaining_features.remove(new_feature)

    print(f'Selected features: {selected_features}')
    sys.stdout.flush()
    return best_model

def train_linear(target_name, dft, dfv, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None):
    '''
    Performs a linear regression with a fixed feature set, or by selecting
    features recursively.

    Returns:
        List of the selected features and the regression coefficients.
    '''
    model = None
    if features is None: # select features with validation data.
        model = linreg_select_features(target_name, dft, dfv, metric,
                feature_groups=feature_groups,
                verbose=verbose, debug_nfeature=debug_nfeature)
    else: # feature set is fixed.
        model = linreg(target_name, dft, features, verbose)
    return model

## Tree fitting

def lgbreg(target_name, dft, dfv, features, max_depth=None, num_leaves=None,
        min_child_samples=None, n_estimators=40, verbose=False):
    X = dft.loc[dft.valid, features]
    y = dft.loc[dft.valid, target_name]
    Xv = dfv.loc[dfv.valid, features]
    yv = dfv.loc[dfv.valid, target_name]

    verbosity = 1 if verbose else -1
    reg_par = {
        'metric': 'rmse',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'verbosity': verbosity,
    }
    fit_par = {
        'eval_set': [(X, y), (Xv, yv)],
        'eval_names': ['train', 'valid'],
        'callbacks': [lgb.early_stopping(stopping_rounds=3, verbose=0), lgb.log_evaluation(period=0)],
    }
    model = lgb.LGBMRegressor(**reg_par)
    model.fit(X, y, **fit_par)
    return model

def lgbreg_select_features(target_name, dft, dfv, metric='rmse', feature_groups=None,
        verbose=False, debug_nfeature=None, min_data_cnt=10):
    '''
    Select feaures recursively using validation sample.

    Returns:
        Series of selected features and the regression coefficients.
    '''
    model = None
    if (dft is None or dfv is None or dft.loc[dft.valid].shape[0] < min_data_cnt
        or dfv.loc[dfv.valid].shape[0] < min_data_cnt):
        return model

    if feature_groups is None:
        feature_groups=['diff_ret', 'medqimb', 'qimax', 'hilo']

    allfeatures = [x for x in dft.columns if np.any([x.startswith(g+'_') for g in feature_groups])]
    if not debug_nfeature is None:
        allfeatures = allfeatures[:min(len(allfeatures), debug_nfeature)]

    print(f'Total {len(allfeatures)} features: {allfeatures}')

    n_rep = 18
    n_sim = 100
    selection_early_stop = True

    max_best_va = 0
    mxd_range = [2, 7]
    nl_range = [10, 100]
    mcs_exp_range = [6, 14]

    final_model = None
    va_list = []
    score_list = []
    dflist = []
    features = allfeatures.copy()
    for irep in range(n_rep):
        if verbose:
            print(f'Starting.. features={features}')
            print(f'mxd: {mxd_range} nl: {nl_range} mcs: {mcs_exp_range}')

        for _ in range(n_sim):
            mxd = np.random.randint(*mxd_range)
            nl = np.random.randint(*nl_range)
            mcs = int(2**np.random.uniform(*mcs_exp_range))
            model = lgbreg(target_name, dft, dfv, features,
                    max_depth=mxd, num_leaves=nl, min_child_samples=mcs, verbose=verbose)

            Xv = dfv.loc[dfv.valid, features]
            yv = dfv.loc[dfv.valid, target_name]
            valid_score = model.score(Xv, yv)
            dfimportance = pd.DataFrame({'name': model.feature_name_, 'importance': model.feature_importances_})
            score_list.append([mxd, nl, mcs, valid_score, dfimportance, model])

        df0 = pd.DataFrame(data=score_list, columns=['mxd', 'nl', 'mcs', 'va', 'imp', 'model'])
        if irep % 2 == 0:
            df1 = df0.loc[df0.va >= df0.va.quantile(0.9)]
            mxd_range = [max(2, int(df1.mxd.min()) - 1), int(df1.mxd.max()) + 2]
            nl_range = [max(2, int(df1.nl.min()) - 1), int(df1.nl.max()) + 2]
            mcs_exp_range = [int(np.log2(df1.mcs.min())) - 1, int(np.log2(df1.mcs.max())) + 1]
        else:
            dflist.append(df0)
            df0 = df0.sort_values(by='va')
            best_row = df0.iloc[-1]
            best_va = best_row.va
            va_list.append(best_va)
            if verbose:
                print(f'best n: {best_row.n} mxd: {best_row.mxd} nl: {best_row.nl} mcs: {best_row.mcs} va: {best_row.va:.4}')

            if best_va > max_best_va:
                max_best_va = best_va;
            elif selection_early_stop:
                break

            model0 = df0.iloc[-1]['model']
            final_model = model0

            dfimp = df0.iloc[-1]['imp']
            dfimp = dfimp.sort_values(by='importance')
            features = dfimp.iloc[len(dfimp)//3:]['name'].tolist()
            score_list = []

    if verbose:
        print(f'Selected features: {selected_features}')
    return final_model

def train_tree(target_name, dft, dfv, metric='rmse', feature_groups=None,
                 features=None, verbose=False, debug_nfeature=None):
    '''
    Performs a boosted tree regression with a fixed feature set, or by selecting
    features recursively.

    Returns:
        List of the selected features and the regression coefficients.
    '''
    selected_features = None
    model = None
    if features is None: # select features with validation data.
        model = lgbreg_select_features(target_name, dft, dfv, metric,
                feature_groups=feature_groups,
                verbose=verbose, debug_nfeature=debug_nfeature)
    else:
        return None
    return model

def get_dts(st, fit_window, val_window, oos_window):
    '''
    Calculate the rolling fitting windows.

    Returns:
        First day of validation, first day of out-of-sample test, and the last day (exclusive)
        of out-of-sample test.
    '''
    dtv = st + timedelta(hours=fit_window)
    dto = dtv + timedelta(hours=val_window)
    dte = dto + timedelta(hours=oos_window)
    return dtv, dto, dte

def oos_linear(par, fitpar, dtt, dtv, dto, dte, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, do_write_pred=True):
    return oos(par, fitpar, dtt, dtv, dto, dte, fit_func=train_linear, metric=metric, feature_groups=feature_groups,
            features=features, verbose=verbose, debug_nfeature=debug_nfeature, do_write_pred=do_write_pred)

def oos_tree(par, fitpar, dtt, dtv, dto, dte, metric='rmse', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, do_write_pred=True):
    return oos(par, fitpar, dtt, dtv, dto, dte, fit_func=train_tree, metric=metric, feature_groups=feature_groups,
            features=features, verbose=verbose, debug_nfeature=debug_nfeature, do_write_pred=do_write_pred)

def oos(par, fitpar, dtt, dtv, dto, dte, fit_func, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, do_write_pred=True):
    '''
    Performs fitting with the specified train, validate, and out-of-sample dates.

    Returns:
        Series of predictions.
    '''
    feature_dir = fitpar['feature_dir']
    dft = read_features(dtt, dtv, feature_dir)
    dfv = read_features(dtv, dto, feature_dir)
    dft['TAR'] = dft[fitpar['target_name']]
    dfv['TAR'] = dfv[fitpar['target_name']]

    # if prior fit exists, read the pred, subtract from target
    if 'prior_fit' in fitpar:
        pred_dir = get_pred_dir_from_name(par, fitpar['prior_fit'])
        train_prior_pred = read_pred_from_dir(pred_dir, dtt, dtv)['totpred']
        valid_prior_pred = read_pred_from_dir(pred_dir, dtv, dto)['totpred']
        dft['TAR'] -= train_prior_pred
        dfv['TAR'] -= valid_prior_pred

    model = fit_func(
            'TAR', dft, dfv, metric=metric, feature_groups=feature_groups,
            features=features, verbose=verbose, debug_nfeature=debug_nfeature)
    selected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.feature_names_ if hasattr(model, 'feature_names_') else None

    if model is not None:
        dfo = read_features(dto, dte, feature_dir, list(selected_features) + [fitpar['target_name']])
        if dfo is not None and dfo.shape[0] > 0:
            Xo = dfo[selected_features].copy()
            pred = pd.Series(model.predict(Xo), index=Xo.index)

            oos_target = dfo[fitpar['target_name']].rename('target')
            if 'prior_fit' in fitpar:
                oos_prior_pred = read_pred_from_dir(pred_dir, dto, dte)['totpred']
                dfo = pd.concat([oos_target,
                    (oos_target - oos_prior_pred).rename('restarget'),
                    pred.rename('respred'),
                    (oos_prior_pred + pred).rename('totpred')], axis=1)
            else:
                dfo = pd.concat([oos_target, pred.rename('totpred')], axis=1)

            if do_write_pred:
                write_pred(dfo, par, fitpar)
            return dfo
    return None

def rolling_oos(par, fitpar, st, et, metric='r2', feature_groups=None,
        features=None, debug_nfeature=None, oos_func=oos_linear):
    '''
    Performs fitting with rolling window between st and et.

    Returns:
        Out of sample prediction.
    '''
    dfo_list = []
    dtt = st
    dtv, dto, dte = get_dts(st, fitpar['fit_window'], fitpar['val_window'], fitpar['oos_window'])
    while(dte <= et):
        print(dtt, dtv, dto, dte)
        sys.stdout.flush()
        dfo = oos_func(par, fitpar, dtt, dtv, dto, dte, metric=metric, feature_groups=feature_groups,
                features=features, debug_nfeature=debug_nfeature)
        if dfo is not None:
            dfo_list.append(dfo)

        dtt = dtt + timedelta(hours=fitpar['oos_window'])
        dtv, dto, dte = get_dts(dtt, fitpar['fit_window'], fitpar['val_window'], fitpar['oos_window'])
    if len(dfo_list) > 0:
        dfoall = pd.concat(dfo_list)
        return dfoall
    return None

def get_pred_dir_from_name(par, fit_name):
    pred_dir = f'{get_fit_dir(par)}/{fit_name}/pred'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    return pred_dir

def get_pred_dir(par, fitpar):
    fit_name = fitpar['target_name']
    if 'fit_desc' in fitpar and fitpar['fit_desc'] != '':
        fit_name += '.' + fitpar['fit_desc']

    pred_dir = get_pred_dir_from_name(par, fit_name)
    return pred_dir

def get_pred_path(dt1, dt2, par, fitpar):
    pred_dir = get_pred_dir(par, fitpar)
    path = f'{pred_dir}/pred.{get_idate(dt1)}.{get_idate(dt2)}.parquet'
    return path

def write_pred(dfo, par, fitpar):
    path = get_pred_path(dfo.index[0], dfo.index[-1], par, fitpar)
    dfo.to_parquet(path)
    print(f'oos pred written to {path}')
    return

def read_pred(par, fitpar, st, et):
    return read_pred_from_dir(get_pred_dir(par, fitpar), st, et)

def read_pred_from_dir(pred_dir, st, et):
    idate1 = get_idate(st)
    idate2 = get_idate(et) # exclusive
    filenames = glob.glob(pred_dir+'/*')
    filenames.sort()
    def within_range(x):
        x = os.path.basename(x)
        xsp = x.split('.')
        if len(xsp) == 4 and xsp[0] == 'pred':
            d1, d2 = xsp[1:3]
            ret = d1 == d2 and d1.isnumeric() and d2.isnumeric() and int(d1) >= idate1 and int(d2) < idate2
            return ret
        return False
    filenames = [x for x in filenames if within_range(x)]
    dflist = []
    for filename in filenames:
        df = pd.read_parquet(filename)
        dflist.append(df)
    dfo = pd.concat(dflist)
    if dfo.index.duplicated().any():
        print('duplicated index')
        return None
    return dfo

def plot_target_prediction(dfo):
    plt.figure(figsize=(12,4))
    ax = plt.subplot(1, 2, 1)
    plot_target_prediction_nquantiles(dfo, n=100, ax=ax)
    ax = plt.subplot(1, 2, 2)
    plot_target_prediction_errorbar(dfo, ax=ax)
    plt.tight_layout()

def plot_target_prediction_nquantiles(dfo, n=100, ax=None):
    '''
    Plots target vs prediciton in 100 prediction quantiles.
    '''
    if ax is None:
        plt.figure()
    corr = dfo.totpred.corr(dfo.target)
    qc = pd.qcut(dfo.totpred, n, duplicates='drop')
    tarpred = dfo.groupby(qc).target.mean()
    mean_preds = dfo.groupby(qc).totpred.mean()
    plt.plot(mean_preds, tarpred, label=f'corr {corr:.4f}')
    plt.title('target vs prediction (out-of-sample)')
    plt.xlabel('prediciton')
    plt.ylabel('target')
    plt.grid()
    plt.legend()

def plot_target_prediction_errorbar(dfo, ax=None):
    '''
    Plots target vs prediction in 13 quantiles of varying sizes.
    '''
    if ax is None:
        plt.figure()
    bc = pd.cut(dfo.totpred, dfo.totpred.quantile([0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99, 1]),
               duplicates='drop')
    bcgrp = dfo.groupby(bc).target.agg(['mean', 'std'])
    bcx = dfo.groupby(bc).totpred.mean()
    plt.errorbar(bcx, bcgrp['mean'], yerr=bcgrp['std'], fmt='o', capsize=5)
    plt.title('target vs prediction w/ error bars')
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.grid()
    plt.tight_layout()

## Trade

def get_pnl(dfo, target_name, feebp):
    '''
    Calculates pnl from prediction following simple rules. Vectorized for speed.
    Any change to this function should be rigorously tested.

    Returns:
        Series of pnl and position.
    '''
    # entry
    fee = feebp * 1e-4
    cost = fee + .5*dfo.adj_width
    entry = dfo.valid & ((dfo.pred > cost) | (dfo.pred < -cost))

    # position
    position = pd.Series(np.nan, index=dfo.index)
    position[entry] = np.sign(dfo.pred)
    position = position.copy().ffill()

    # exit condition
    exitcond = (~dfo.valid) | (~entry & (((position.shift() > 0) & (dfo.pred < -cost)) | ((position.shift() < 0) & (dfo.pred > cost))))
    position[exitcond] = 0

    # group and transform
    dummy = entry | exitcond
    position = position.groupby(dummy.cumsum()).transform(lambda x: x.iloc[0]).fillna(0)
    pnl = (position * dfo[target_name] - (position.shift() - position).abs()*cost).fillna(0)

    pnl.name = (feebp, 'pnl')
    position.name = (feebp, 'pos')
    entry.name = (feebp, 'entry')
    return pnl, position, entry

def get_aggpnl(pnl):
    '''
    Downsamples the pnl series.

    Returns:
        A pnl series.
    '''
    aggpnl = None
    time_range = pnl.index[-1] - pnl.index[0]
    if time_range > timedelta(days=100):
        aggpnl = pnl.groupby(pnl.index.to_series().apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))).sum()
    elif time_range > timedelta(hours=100):
        aggpnl = pnl.groupby(pnl.index.to_series().apply(lambda x: x.replace(minute=0, second=0, microsecond=0))).sum()
    else:
        aggpnl = pnl.groupby(pnl.index.to_series().apply(lambda x: x.replace(second=0, microsecond=0))).sum()

    return aggpnl

def get_pnls(dfo, target_name, feebp_list):
    '''
    Calculates multiple pnl series using the trading fees passed from the caller.

    Returns:
        Dataframe with a multiindex column. The first level of the multiindex column is
        the trading fee in basis points. The second level is ('pnl', 'position').
    '''
    cols = []
    aggcols = []
    for feebp in feebp_list:
        pnl, pos, entry = get_pnl(dfo, target_name, feebp)
        cols.append(pnl)
        cols.append(pos)
        cols.append(entry)

        aggpnl = get_aggpnl(pnl)
        aggcols.append(aggpnl)
    df = pd.concat(cols, axis=1)
    dfagg = pd.concat(aggcols, axis=1)
    return df, dfagg

def get_daily_sharpe(pnl):
    '''
    Calculates daily sharpe from pnl series of arbitrary time interval.
    If the input interval is too small, aggregate the series before calculating the sharpe.

    Returns:
        A sharpe ratio.
    '''
    d_shrp = 0
    aggpnl = get_aggpnl(pnl)
    pnlstd = aggpnl.std()
    if pnlstd > 0:
        dfac = (timedelta(days=1).total_seconds() / (aggpnl.index[1]-aggpnl.index[0]).total_seconds())**.5
        pnlmean = aggpnl.mean()
        d_shrp = pnlmean / pnlstd * dfac
    return d_shrp

def plot_pnl(dfpnl):
    '''
    Plots the cumulative pnl.
    '''
    plt.figure()
    feebp_list = dfpnl.columns.get_level_values(0).unique()
    for feebp in feebp_list:
        pnl = dfpnl[feebp]['pnl']
        plt.plot(pnl.cumsum(), label=f'fee={feebp}bp')
    plt.title('cumulative pnl')
    plt.ylabel('pnl')
    plt.xticks(rotation=20)
    plt.grid()
    plt.legend()

def get_trade_summary(dfpnl, dfo=None, target_name=None):
    '''
    Calculate stats of the trading.

    Returns:
        A dataframe.
    '''
    biaspct = 0 # A measure of overfitting. Indpendent of trading cost.
    if dfo is not None and target_name in dfo.columns:
        predlim = dfo.pred.quantile([0.01, 0.99])
        dftopbot = dfo[(dfo.pred < predlim.iloc[0]) | (dfo.pred > predlim.iloc[1])]
        biaspct = ((dftopbot.pred - dftopbot[target_name]) * np.sign(dftopbot.pred)).mean() / dftopbot.pred.abs().mean()

    summ_list = []
    feebp_list = dfpnl.columns.get_level_values(0).unique()
    for feebp in feebp_list:
        df = dfpnl[feebp]
        pos = df['pos']
        pnl = df['pnl']
        entry = df['entry']

        ndays = (pos.index[-1] - pos.index[0]).total_seconds()/60/60/24

        n_data_points = pos.shape[0] # number of data points
        n_nan = pos[pos.isna()].shape[0] # Nan's in pos
        n_nopos = pos[pos==0].shape[0] # no position

        n_take = pos[(pos!=0)&(pos.shift()!=pos)].shape[0] / ndays # entries
        n_exit = pos[(pos==0)&(pos.shift()!=pos)&(pos.shift().notna())].shape[0] / ndays # exits
        n_flip = pos[(pos.shift()*pos<0)].shape[0]/ ndays # flips

        n_pos = pos.mean()
        g_pos = pos.abs().mean()
        sample_interval = (pos.index[1]-pos.index[0]).total_seconds()
        dfpos = pos.groupby((pos!= pos.shift()).cumsum()).agg(len=('count'), val=('first'))
        holding = dfpos.loc[dfpos.val!=0, 'len'].mean() * sample_interval # average holding
        median_holding = dfpos.loc[dfpos.val!=0, 'len'].median() * sample_interval # median holding

        d_volume = (pos - pos.shift()).abs().sum()/ndays # daily volume
        d_pnl = pnl.sum() / ndays
        d_shrp = get_daily_sharpe(pnl)

        sig_rat = entry[entry].sum() / len(entry)

        dft = df.groupby((df.pos.shift() != df.pos).cumsum()).agg(
            pos=('pos', 'mean'), pnl=('pnl', lambda x: x.sum()))
        dft = dft[dft.pos != 0]

        w_rat = len(dft[dft.pnl > 0]) / len(dft) if len(dft) > 0 else 0
        dftb = dft[dft.pos > 0]
        bw_rat = len(dftb[dftb.pnl > 0]) / len(dftb) if len(dftb) > 0 else 0
        dfts = dft[dft.pos < 0]
        sw_rat = len(dfts[dfts.pnl > 0]) / len(dftb) if len(dftb) > 0 else 0

        gpt = dft.pnl.mean()*1e4
        b_gpt = dft[dft.pos > 0].pnl.mean()*1e4
        s_gpt = dft[dft.pos < 0].pnl.mean()*1e4
        (w_mean, w_std) = dft[dft.pnl>0].pnl.agg(['mean', 'std'])*1e4
        (l_mean, l_std) = dft[dft.pnl<0].pnl.agg(['mean', 'std'])*1e4

        mbiaspct = 0 # A measure of overfitting for marketable sample.
        if dfo is not None and target_name in dfo.columns:
            dfentry = dfo[dfo.valid & (pos.shift() != pos) & (pos != 0)]
            mbiaspct = ((dfentry.pred - dfentry[target_name]) * np.sign(dfentry.pred)).mean() / dfentry.pred.abs().mean()

        summ = dict(
            fee = round(feebp, 2),
            n_take = round(n_take, 1),
            n_exit = round(n_exit, 1),
            n_flip = round(n_flip, 1),

            n_pos = round(n_pos, 4),
            g_pos = round(g_pos, 4),
            holding = round(holding, 1),
            bias = round(biaspct, 2),
            mbias = round(mbiaspct, 2),
            d_volume = round(d_volume, 1),
            d_pnl = round(d_pnl, 3),
            d_shrp = round(d_shrp, 2),

            sig_rat = round(sig_rat, 2),
            w_rat = round(w_rat, 2),
            bw_rat = round(bw_rat, 2),
            sw_rat = round(sw_rat, 2),
            gpt = round(gpt, 2),
            b_gpt = round(b_gpt, 2),
            s_gpt = round(s_gpt, 2),
            w_mean = round(w_mean, 2),
            w_std = round(w_std, 2),
            l_mean = round(l_mean, 2),
            l_std = round(l_std, 2),
         )
        summ_list.append(summ)

    dftsumm = pd.DataFrame(summ_list).set_index('fee')
    return dftsumm

def basic_cols():
    return ['n_take', 'n_exit', 'n_flip', 'n_pos', 'g_pos', 'holding', 'bias', 'mbias',
       'd_volume', 'd_pnl', 'd_shrp']

def detail_cols():
    return ['sig_rat', 'w_rat', 'bw_rat', 'sw_rat', 'gpt', 'b_gpt', 's_gpt', 'w_mean', 'w_std',
       'l_mean', 'l_std']

def print_markdown(dftrdsumm):
    '''
    Prints out the formatted text to be included in the markdown document as a table.
    '''
    print('|fee|', '|'.join(dftrdsumm.columns), '|')
    print('|---|', '|'.join(['--:'] * len(dftrdsumm.columns)), '|')
    for indx, row in dftrdsumm.iterrows():
        print('|', indx, '|', '|'.join(row.astype(str).tolist()), '|')
