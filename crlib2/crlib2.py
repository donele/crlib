import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import multiprocessing as mp
import os
import sys
import glob
import pickle
import seaborn as sns
import yaml
from datetime import timedelta
import pyarrow.parquet as pq

from .crdataio import *
from .crfeatures import *
from .crlinreg import *
from .crpaths import *
from .crtreereg import *

dfuniv = None
upath = 'universe.csv'
if os.path.exists(upath):
    dfuniv = pd.read_csv(upath, index_col=[0, 1])

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

### Features

def get_features_1h(dt, par, min_timex=None, max_timex=None, tevt_max_row=18000):
    df = get_features_3h(dt, par, min_timex, max_timex)
    if df is not None:
        dt2 = dt + timedelta(hours=1, microseconds=-1)
        df = df.loc[dt:dt2]
        if 'sample_type' in par and par['sample_type'] == 'tevt':
            df = df.iloc[np.unique(np.arange(0, len(df)-1, len(df)/tevt_max_row).astype(int))]
    return df

def write_feature(dt, par):
    df0 = None
    df0 = get_features_1h(dt, par)
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

def read_features(dt1, dt2, feature_dir, columns=None, feature_groups=None):
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

        available_columns = pq.read_table(path).schema.names
        read_columns = []
        default_columns = ['mid', 'adj_width', 'valid', 'tsince_trade', 'tlat', 'blat']
        read_columns.extend([x for x in default_columns if x in available_columns])
        if columns is not None:
            read_columns.extend([x for x in columns if x in available_columns])
        if feature_groups is not None:
            for g in feature_groups:
                read_columns.extend([x for x in available_columns if x.startswith(g)])
        read_columns = list(set(read_columns))

        df = pd.read_parquet(path, columns=read_columns)
        if df is not None and df.shape[0] > 0:
            dflist.append(df)
    if len(dflist) > 0:
        df = pd.concat(dflist)
        return df
    return None

def plot_test_features(dff, feature_group=None):
    tarcols = [x for x in dff.columns if x.startswith('tar')]
    if feature_group is None:
        ncol = len(tarcols)
        nx = int(ncol**.5) + 1
        ny = nx + 1

        dff[tarcols].hist(figsize=(16,2*ny), bins=80)
        dff[tarcols].hist(figsize=(16,2*ny), bins=80, log=True)

    else:
        cols = [x for x in dff.columns if x.startswith(feature_group+'_')]
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

### Fitting

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

def oos(par, fitpar, dtt, dtv, dto, dte, fit_func, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, do_write_pred=True):
    '''
    Performs fitting with the specified train, validate, and out-of-sample dates.

    Returns:
        Series of predictions.
    '''
    feature_dir = fitpar['feature_dir'] if 'feature_dir' in fitpar else get_feature_dir(par)
    dft = read_features(dtt, dtv, feature_dir, columns=[fitpar['target_name']], feature_groups=feature_groups)
    dfv = read_features(dtv, dto, feature_dir, columns=[fitpar['target_name']], feature_groups=feature_groups)
    if dft is None or dfv is None or len(dft) < 1 or len(dfv) < 1:
        return None

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
    del dft
    del dfv
    selected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.feature_names_ if hasattr(model, 'feature_names_') else None

    if model is not None:
        dfo = read_features(dto, dte, feature_dir, columns=list(selected_features) + [fitpar['target_name']])
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
            write_model(model, dto, par, fitpar)
        del dfo

def rolling_oos(par, fitpar, st, et, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, oos_func=None):
    '''
    Performs fitting with rolling window between st and et.

    Returns:
        Out of sample prediction.
    '''
    dtt = st
    dtv, dto, dte = get_dts(st, fitpar['fit_window'], fitpar['val_window'], fitpar['oos_window'])
    while(dte <= et):
        print(f'train: {dtt}-, validate: {dtv}-, oos: {dto}-{dte}')
        sys.stdout.flush()
        oos_func(par, fitpar, dtt, dtv, dto, dte, metric=metric, feature_groups=feature_groups,
                features=features, verbose=verbose, debug_nfeature=debug_nfeature)

        dtt = dtt + timedelta(hours=fitpar['oos_window'])
        dtv, dto, dte = get_dts(dtt, fitpar['fit_window'], fitpar['val_window'], fitpar['oos_window'])
    return None

def oos_linear(par, fitpar, dtt, dtv, dto, dte, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, do_write_pred=True):
    if feature_groups is None:
        feature_groups=['ret', 'medqimb', 'qimax', 'hilo', 'twret'],
    return oos(par, fitpar, dtt, dtv, dto, dte, fit_func=train_linear, metric=metric, feature_groups=feature_groups,
            features=features, verbose=verbose, debug_nfeature=debug_nfeature, do_write_pred=do_write_pred)

def oos_tree(par, fitpar, dtt, dtv, dto, dte, metric='rmse', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, do_write_pred=True):
    if feature_groups is None:
        feature_groups=['ret', 'medqimb', 'qimax', 'hilo', 'twret', 'diff_sum_net_qty']
    return oos(par, fitpar, dtt, dtv, dto, dte, fit_func=train_tree, metric=metric, feature_groups=feature_groups,
            features=features, verbose=verbose, debug_nfeature=debug_nfeature, do_write_pred=do_write_pred)

def rolling_oos_linear(par, fitpar, st, et, metric='r2',
        feature_groups=['ret', 'medqimb', 'qimax', 'hilo', 'twret'],
        features=None, verbose=False, debug_nfeature=None):
    '''
    Repeat the linear regression with the sliding windows.

    If the process takes more than half the available memory, and the garbage collector doesn't
    free the data early enough, the job may crash.
    '''
    rolling_oos(par, fitpar, st, et, metric=metric, feature_groups=feature_groups,
        features=features, verbose=verbose, debug_nfeature=debug_nfeature, oos_func=oos_linear)

def rolling_oos_tree(par, fitpar, st, et, metric='r2',
        feature_groups=['ret', 'medqimb', 'qimax', 'hilo', 'twret', 'diff_sum_net_qty'],
        features=None, verbose=False, debug_nfeature=None):
    '''
    Repeat the boosted tree regression with the sliding windows.

    If the process takes more than half the available memory, and the garbage collector doesn't
    free the data early enough, the job may crash.
    '''
    rolling_oos(par, fitpar, st, et, metric=metric, feature_groups=feature_groups,
        features=features, verbose=verbose, debug_nfeature=debug_nfeature, oos_func=oos_tree)

def write_pred(dfo, par, fitpar):
    path = get_pred_path(dfo.index[0], dfo.index[-1], par, fitpar)
    dfo.to_parquet(path)
    print(f'oos pred written to {path}')

def write_model(model, dt, par, fitpar):
    path = get_model_path(dt, par, fitpar)
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def read_pred(par, st, et, target_name, fit_desc=''):
    return read_pred_from_dir(get_pred_dir(par, target_name, fit_desc), st, et)

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

def read_oos(st, et, par, predpar):
    targets = predpar['target_names'] if 'target_names' in predpar else [predpar['target_name']]
    descs = predpar['fit_descs'] if 'fit_descs' in predpar else [predpar['fit_desc']]
    weights = predpar['weights'] if 'weights' in predpar else [1] * len(targets)

    feature_dir = predpar['feature_dir'] if 'feature_dir' in predpar else get_feature_dir(par)
    read_targets = list(set(targets + [predpar['target_name']]))
    dfo = read_features(st, et, feature_dir, columns=['mid', 'adj_width', 'valid', 'tsince_trade', 'tlat', 'blat'] + read_targets)

    pred_list = []
    for t, d in zip(targets, descs):
        dfp = read_pred(par, st, et, t, d)
        pred = dfp.totpred
        pred.name = t.replace('tar', 'pred')
        pred_list.append(pred)
    sumpred = (pd.concat(pred_list, axis=1) * weights).sum(axis=1)
    sumpred.name = 'pred'
    dfo = pd.concat([dfo] + pred_list + [sumpred], axis=1)

    return dfo

### Prediction Plots

def plot_target_prediction(dfo, target_name='target', pred_col='pred', nq=20,
        yrange_noerr=None, yrange_err=None):
    plt.figure(figsize=(12,4))
    ax = plt.subplot(1, 2, 1)
    plot_target_prediction_nquantiles(dfo, target_name, pred_col, nq=nq, ax=ax, yrange=yrange_noerr)
    ax = plt.subplot(1, 2, 2)
    plot_target_prediction_errorbar(dfo, target_name, pred_col, ax=ax, yrange=yrange_err)
    plt.tight_layout()

def plot_target_prediction_nquantiles(dfo, target_name='target', pred_col='pred', nq=100, ax=None, yrange=None):
    '''
    Plots target vs prediciton in 100 prediction quantiles.
    '''
    if ax is None:
        plt.figure()
    corr = dfo[pred_col].corr(dfo[target_name])
    qc = pd.qcut(dfo[pred_col], nq, duplicates='drop')
    tarpred = dfo.groupby(qc)[target_name].mean()
    mean_preds = dfo.groupby(qc)[pred_col].mean()
    plt.plot(mean_preds, tarpred, label=f'corr {corr:.4f}')
    if yrange is not None:
        plt.ylim(*yrange)
    plt.title('target vs prediction (out-of-sample)')
    plt.xlabel('prediciton')
    plt.ylabel('target')
    plt.grid()
    plt.legend()

def plot_target_prediction_errorbar(dfo, target_name='target', pred_col='pred', ax=None, yrange=None):
    '''
    Plots target vs prediction in 13 quantiles of varying sizes.
    '''
    if ax is None:
        plt.figure()
    bc = pd.cut(dfo[pred_col], dfo[pred_col].quantile([0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99, 1]),
               duplicates='drop')
    bcgrp = dfo.groupby(bc)[target_name].agg(['mean', 'std'])
    bcx = dfo.groupby(bc)[pred_col].mean()
    plt.errorbar(bcx, bcgrp['mean'], yerr=bcgrp['std'], fmt='o', capsize=5)
    if yrange is not None:
        plt.ylim(*yrange)
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
        pnl: pd.Series (required for trade_summary())
        pos: pd.Series (required for trade_summary())
        entry: pd.Series (optional for trade_summary())
    '''
    # entry
    fee = feebp * 1e-4
    cost = fee + .5*dfo.adj_width
    entry = dfo.valid & ((dfo.pred > cost) | (dfo.pred < -cost))

    # position
    pos0 = pd.Series(np.nan, index=dfo.index)
    pos0[entry] = np.sign(dfo.pred)
    pos0 = pos0.copy().ffill()

    # exit condition
    exitcond = (~dfo.valid) | (~entry & (((pos0.shift() > 0) & (dfo.pred < -cost)) | ((pos0.shift() < 0) & (dfo.pred > cost))))
    pos0[exitcond] = 0

    # group and transform for final position series
    dummy = entry | exitcond
    pos = pd.Series(np.nan, index=dfo.index)
    pos.loc[dummy] = pos0.loc[dummy]
    pos = pos.ffill().fillna(0)

    # pnl
    pnlcarry = (pos * (dfo.mid.shift(-1)/dfo.mid - 1).fillna(0)).shift()
    pnlinsert = - (pos.shift() - pos).fillna(0).abs()*cost
    pnl = pnlcarry + pnlinsert

    pnl.name = (feebp, 'pnl')
    pos.name = (feebp, 'pos')
    entry.name = (feebp, 'entry')
    return pnl, pos, entry

class PredRange:
    def __init__(self, dfo, target_name, ticksize=0.1, nq=20):
        self.ticksize = ticksize
        qc = pd.qcut(dfo.pred, nq)
        gr = dfo.groupby(qc)[target_name].agg(['mean', 'std'])
        grx = dfo.groupby(qc).pred.mean()
        self.coef = np.polyfit(grx, gr['std'], 2)
    def get_std(self, pred):
        c = self.coef
        std = c[0] * pred**2 + c[1] * pred + c[2]
        return std
    def pred_range(self, pred, p, q, r):
        d = (q + r) * self.get_std(pred)
        return p * pred - d, p * pred + d
    def buy_sell_prc(self, mid, pred, p, q, r):
        pred_range = self.pred_range(pred, p, q, r)
        buyprc = min(mid - self.ticksize, mid * (1 + pred_range[0]))
        sellprc = max(mid + self.ticksize, mid * (1 + pred_range[1]))
        return buyprc, sellprc

def get_mm_pnl(dfo, target_name, rebatebp=0.5, p=1, q=1, r=0, pr=None):
    rebate = rebatebp * 1e-4
    if pr is None:
        pr = PredRange(dfo, target_name)
    dfmm = pd.DataFrame({'mid': 0, 'pos': 0, 'buyprc': 0, 'sellprc': 1e9, 'pred': 0, 'uppred': 0, 'dnpred': 0,
        'pnlcarry': 0, 'pnlbuy': 0, 'pnlsell': 0, 'rebate': 0, 'pnl': 0}, index=dfo.index)

    buysize = 0
    buyprc = 0
    sellsize = 0
    sellprc = 1e9
    currpos = 0
    for idx, row in dfo.iterrows():
        pred = row.pred
        pred_range = pr.pred_range(pred, p, q, r)
        uppred = pred_range[0]
        dnpred = pred_range[1]

        if buysize > 0 and buyprc > row.mid: # Buy Fill
            currpos += buysize
            dfmm.loc[idx, 'pnlbuy'] = (row.mid / buyprc - 1)
            buysize = 0
            buyprc = 0
        if sellsize > 0 and sellprc < row.mid: # Sell Fill
            currpos -= sellsize
            dfmm.loc[idx, 'pnlsell'] = -(row.mid / sellprc - 1)
            sellsize = 0
            sellprc = 0

        if row.tlat > 10000: # Cancel if trade latency > 10ms
            buysize = 0
            buyprc = 0
            sellsize = 0
            sellprc = 0
        else:
            buyprc0, sellprc0 = pr.buy_sell_prc(row.mid, pred, p, q, r)
            prcok = buyprc0 < sellprc0

            if prcok and currpos <= 0: # Buy Order
                buysize = 1
                buyprc = buyprc0
            else:
                buysize = 0
                buyprc = 0

            if prcok and currpos >= 0: # Sell Order
                sellsize = 1
                sellprc = sellprc0
            else:
                sellsize = 0
                sellprc = 0

        dfmm.loc[idx, ['mid', 'pos', 'buyprc', 'sellprc']] = [row.mid, currpos, buyprc, sellprc]
        dfmm.loc[idx, ['pred', 'uppred', 'dnpred']] = [pred*1e4, uppred*1e4, dnpred*1e4]

    dfmm.pnlcarry = (dfmm.pos * (dfo.mid.shift(-1) / dfo.mid - 1).fillna(0)).shift()
    dfmm.rebate = rebate * (dfmm.pos.shift() - dfmm.pos).fillna(0).abs()
    dfmm.pnl = dfmm.pnlcarry + dfmm.rebate + dfmm.pnlbuy + dfmm.pnlsell
    dfmm.columns = pd.MultiIndex.from_product([[q], dfmm.columns], names=['q', ''])
    return dfmm

def get_aggpnl(pnl):
    '''
    Downsamples the pnl series.

    Returns:
        A pnl series.
    '''
    aggpnl = None
    time_range = pnl.index[-1] - pnl.index[0]
    if time_range > timedelta(days=100):
        day_in_ns = int(60*60*1e9)
        day_to_us = int(60*60*1e6)
        aggpnl = pnl.groupby(pd.to_datetime(pnl.index.astype(int)//day_in_ns*day_to_us, unit='us')).sum()
    elif time_range > timedelta(hours=100):
        hr_in_ns = int(60*60*1e9)
        hr_to_us = int(60*60*1e6)
        aggpnl = pnl.groupby(pd.to_datetime(pnl.index.astype(int)//hr_in_ns*hr_to_us, unit='us')).sum()
    elif time_range > timedelta(minutes=100):
        min_in_ns = int(60*1e9)
        min_to_us = int(60*1e6)
        aggpnl = pnl.groupby(pd.to_datetime(pnl.index.astype(int)//min_in_ns*min_to_us, unit='us')).sum()
    else:
        sec_in_ns = int(1e9)
        sec_to_us = int(1e6)
        aggpnl = pnl.groupby(pd.to_datetime(pnl.index.astype(int)//sec_in_ns*sec_to_us, unit='us')).sum()

    return aggpnl

def get_pnls(dfo, target_name, feebp_list):
    '''
    Calculates multiple pnl series using the trading fees passed from the caller.
    Use the multiprocessing module.

    Returns:
        Dataframe with a multiindex column. The first level of the multiindex column is
        the trading fee in basis points. The second level is ('pnl', 'position').
    '''
    cols = []
    for feebp in feebp_list:
        pnl, pos, entry = get_pnl(dfo, target_name, feebp)
        cols.append(pnl)
        cols.append(pos)
        cols.append(entry)
    df = pd.concat(cols, axis=1)
    return df

def get_mm_pnls_mp(dfo, target_name, rebatebp=0.5, p=1, q_list=[1], r=0, pr=None):
    pnls = []

    pool = mp.Pool(processes=6)
    results = [pool.apply_async(get_mm_pnl, args=(dfo, target_name, rebatebp, p, q, r, pr))
            for q in q_list]
    pool.close()
    pool.join()
    for result in results:
        pnl = result.get()
        if pnl is not None:
            pnls.append(pnl)
    df = pd.concat(pnls, axis=1)
    return df

def get_mm_pnls(dfo, target_name, rebatebp=0.5, p=1, q_list=[1], r=0, pr=None):
    pnls = []
    for q in q_list:
        dfmm = get_mm_pnl(dfo, target_name, rebatebp, p=p, q=q, r=r, pr=pr)
        if dfmm is not None:
            pnls.append(dfmm)
    df = pd.concat(pnls, axis=1)
    return df

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

def plot_pnl(dfpnl, col='pnl', cum=True, label='fee', figsize=None):
    '''
    Plots the cumulative pnl.
    '''
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    feebp_list = dfpnl.columns.get_level_values(0).unique()
    for feebp in feebp_list:
        pnl = dfpnl[feebp][col]
        aggpnl = get_aggpnl(pnl)
        if cum:
            plt.plot(aggpnl.cumsum(), label=f'{label}={feebp}')
        else:
            plt.plot(aggpnl, label=f'{label}={feebp}')
    if cum:
        plt.title(f'cumulative {col}')
    else:
        plt.title(f'{col}')
    plt.ylabel(f'{col}')
    plt.xticks(rotation=20)
    plt.grid()
    plt.legend()

def get_holding_summary(dfpnl, dfo):
    pass

def get_trade_summary_fee(feebp, df, dfo, target_name, label):
    biaspct = 0 # A measure of overfitting. Indpendent of trading cost.
    if dfo is not None and target_name in dfo.columns:
        predlim = dfo.pred.quantile([0.01, 0.99])
        dftopbot = dfo[(dfo.pred < predlim.iloc[0]) | (dfo.pred > predlim.iloc[1])]
        biaspct = ((dftopbot.pred - dftopbot[target_name]) * np.sign(dftopbot.pred)).mean() / dftopbot.pred.abs().mean()

    pos = df['pos'].fillna(0)
    pnl = df['pnl'].fillna(0)
    entry = df['entry'].fillna(False) if 'entry' in df.columns else pd.Series(False, index=pos.index)

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

    volume = (pos - pos.shift()).abs().sum()
    d_volume = volume / ndays # daily volume
    d_pnl = pnl.sum() / ndays
    d_shrp = get_daily_sharpe(pnl)

    sig_rat = entry[entry].sum() / len(entry)

    dft = df.groupby((df.pos.shift() != df.pos).cumsum()).agg(
        pos=('pos', 'mean'), pnl=('pnl', 'sum'))
    dft = dft[dft.pos != 0]

    w_rat = len(dft[dft.pnl > 0]) / len(dft) if len(dft) > 0 else 0
    dftb = dft[dft.pos > 0]
    bw_rat = len(dftb[dftb.pnl > 0]) / len(dftb) if len(dftb) > 0 else 0
    dfts = dft[dft.pos < 0]
    sw_rat = len(dfts[dfts.pnl > 0]) / len(dftb) if len(dftb) > 0 else 0

    gpt = 1e4 * dft.pnl.sum() / volume
    b_gpt = 0# dft[dft.pos > 0].pnl.mean()*1e4
    s_gpt = 0#dft[dft.pos < 0].pnl.mean()*1e4
    (w_mean, w_std) = dft[dft.pnl>0].pnl.agg(['mean', 'std'])*1e4
    (l_mean, l_std) = dft[dft.pnl<0].pnl.agg(['mean', 'std'])*1e4

    mbiaspct = 0 # A measure of overfitting for marketable sample.
    if dfo is not None and target_name in dfo.columns:
        dfentry = dfo[dfo.valid & (pos.shift() != pos) & (pos != 0)]
        mbiaspct = ((dfentry.pred - dfentry[target_name]) * np.sign(dfentry.pred)).mean() / dfentry.pred.abs().mean()

    summ = {
        label: round(feebp, 2),
        'n_take': round(n_take, 1),
        'n_exit': round(n_exit, 1),
        'n_flip': round(n_flip, 1),

        'n_pos': round(n_pos, 4),
        'g_pos': round(g_pos, 4),
        'holding': round(holding, 1),
        'bias': round(biaspct, 2),
        'mbias': round(mbiaspct, 2),
        'd_volume': round(d_volume, 1),
        'd_pnl': round(d_pnl, 3),
        'd_shrp': round(d_shrp, 2),

        'sig_rat': round(sig_rat, 2),
        'w_rat': round(w_rat, 2),
        'bw_rat': round(bw_rat, 2),
        'sw_rat': round(sw_rat, 2),
        'gpt': round(gpt, 4),
        'b_gpt': round(b_gpt, 4),
        's_gpt': round(s_gpt, 4),
        'w_mean': round(w_mean, 2),
        'w_std': round(w_std, 2),
        'l_mean': round(l_mean, 2),
        'l_std': round(l_std, 2),
    }
    return summ

def get_trade_summary(dfpnl, dfo=None, target_name=None, label='fee'):
    '''
    Calculate stats of the trading.

    Returns:
        A dataframe.
    '''
    summ_list = []
    feebp_list = dfpnl.columns.get_level_values(0).unique()
    for feebp in feebp_list:
        df = dfpnl[feebp]
        summ = get_trade_summary_fee(feebp, df, dfo, target_name, label)
        summ_list.append(summ)

    dftsumm = pd.DataFrame(summ_list).set_index(label)
    return dftsumm

def get_trade_summary_mp(dfpnl, dfo=None, target_name=None, label='fee'):
    '''
    Calculate stats of the trading by multiprocessing. Overhead may not justify doing this.

    Returns:
        A dataframe.
    '''
    summ_list = []
    feebp_list = dfpnl.columns.get_level_values(0).unique()

    pool = mp.Pool(processes=6)
    results = [pool.apply_async(get_trade_summary_fee, args=(feebp, dfpnl[feebp], dfo, target_name, label))
            for feebp in feebp_list]
    pool.close()
    pool.join()
    for result in results:
        summ_list.append(result.get())

    dftsumm = pd.DataFrame(summ_list).set_index(label)
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

def steps(df):
    '''
    Manipulates the price series to be used for the step-like representation of the prices.

    Returns:
        Modified price series.
    '''
    index_name = df.index.name
    tmp_index_name = 'tmp'
    shifted = df.shift().dropna()
    return pd.concat([shifted, df]).reset_index().rename_axis(tmp_index_name).sort_values(
        by=[index_name, tmp_index_name]).set_index(index_name)

def plot_trade(dt, span, dft, dfb, dfm):
    '''
    Plots the price series of bbo, trade, and midpx data.
    '''
    dtfrom = dt - timedelta(seconds=span)
    dtto = dt + timedelta(seconds=span)

    fig, ax = plt.subplots(figsize=(12,3))
    fmt = '.-'

    plt.plot(steps(dfb.loc[dtfrom:dtto, 'bidpx']), fmt, label='bid')
    plt.plot(steps(dfb.loc[dtfrom:dtto, 'askpx']), fmt, label='ask')
    plt.plot(steps(dfm.loc[dtfrom:dtto, 'mid_px']), fmt, label='mid')
    plt.plot(dft.loc[dtfrom:dtto].price, '.', label='trade')

    plt.title(f'{dt} +- {span}sec')
    myFmt = mdates.DateFormatter(":%S.%f")
    ax.xaxis.set_major_formatter(myFmt)

    plt.grid()
    plt.legend()

### Vectorbt

def plot_pf(pf):
    plt.figure(figsize=(12,2))
    plt.subplot(121)
    pf.asset_value.plot(title='asset_value')
    plt.subplot(122)
    (pf.value - pf.init_value).plot(title='cum pnl')

def df_from_pf(pf):
    showcols = ['close', 'asset_value', 'asset_flow', 'position', 'returns', 'cumulative_returns']
    showsers = []
    for x in showcols:
        ser = getattr(pf, x)
        ser.name = x
        showsers.append(ser)
    df = pd.concat(showsers, axis=1)
    return df

def get_df_vbt(dfo, serstd, p=1, q=2, r=0, maxlat=1e9):
    serprice = dfo.mid
    buyprc = dfo.mid * (1 - (q + r) * serstd + p * dfo.pred)
    sellprc = dfo.mid * (1 + (q + r) * serstd + p * dfo.pred)
    buyprc[dfo.tlat > maxlat] = 1e-9;
    sellprc[dfo.tlat > maxlat] = 1e9;
    longentry = buyprc.shift() > serprice
    shortentry = sellprc.shift() < serprice
    longslippage = (serprice - buyprc.shift()).abs()
    shortslippage = (serprice - sellprc.shift()).abs()
    slippage = pd.Series(0, index=serprice.index)
    slippage[longentry] = longslippage[longentry]
    slippage[shortentry] = shortslippage[shortentry]
    slippage /= serprice

    dfv = pd.DataFrame({'price': serprice,
        'buyprc': buyprc,
        'sellprc': sellprc,
        'longentry': longentry,
        'shortentry': shortentry,
        'slippage': slippage})
    return dfv

import vectorbtpro as vbt
from vectorbtpro import *
@njit
def signal_func_mm(c, prc, buyprc, sellprc):
    #posthres = (maxpos[c.i] - .5) / prc[c.i]
    posthres = (1 - .5) / prc[c.i]
    pos = c.last_position[0]
    long_entry = buyprc[c.i] > prc[c.i] and pos < posthres
    short_entry = sellprc[c.i] < prc[c.i] and pos > -posthres
    long_exit = pos > 0 and pos < posthres
    short_exit = pos < 0 and pos > -posthres
    return long_entry, long_exit, short_entry, short_exit

def get_mm_pnl_vbt(dfo, target_name, serstd, feebp=-0.5, p=1, q=2, r=0, maxpos=1, maxlat=1e9):
    '''
    Calculates pnl by calling from_signals function of vectorbt.

    Take serstd from the caller.

    Negative fee means rebate.
    '''
    fee = feebp * 1e-4
    dfv = get_df_vbt(dfo, serstd, p, q, r, maxlat)

    pf = vbt.Portfolio.from_signals(dfv.price, fees=fee, slippage=dfv.slippage,
        signal_func_nb=signal_func_mm, signal_args=(
            dfv.price.values,
            dfv.buyprc.shift().values,
            dfv.sellprc.shift().values,
            ),
        accumulate=True, init_cash=maxpos, size=1, size_type='value')

    pnl = (pf.value - pf.init_value).diff().fillna(0)
    pos = pf.asset_value
    dfmm = pd.DataFrame({'pnl': pnl, 'pos': pos})
    dfmm.columns = pd.MultiIndex.from_product([[q], dfmm.columns], names=['q', ''])
    return dfmm

def get_mm_pnls_vbt(dfo, target_name, q_list, feebp=-0.5, maxpos=None, pr=None, serstd=None, maxlat=1e9):
    if serstd is None:
        if pr is None:
            pr = PredRange(dfo, target_name)
        serstd = dfo.pred.apply(lambda x: pr.get_std(x))
    cols = []
    for q in q_list:
        dfmm = get_mm_pnl_vbt(dfo, target_name, serstd, feebp=feebp, q=q, maxpos=maxpos, maxlat=maxlat)
        cols.append(dfmm)
    df = pd.concat(cols, axis=1)
    return df
