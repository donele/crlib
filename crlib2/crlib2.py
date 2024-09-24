import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import glob
import pickle
import datetime
import seaborn as sns
import yaml
from datetime import timedelta
from math import *

from crlib2.crpaths import *
from crlib2.crdataio import *
from crlib2.crfeatures import *
from crlib2.crlinreg import *
from crlib2.crtreereg import *

## Data Hangling

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

def read_oos(st, et, par, fitpar):
    dfp = read_pred(par, fitpar, st, et)
    dfo = read_features(st, et, fitpar['feature_dir'], columns=[
        'mid', 'adj_width', 'valid', 'tsince_trade', fitpar['target_name']])
    dfo['pred'] = dfp.totpred
    return dfo

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
    feature_dir = fitpar['feature_dir']
    dft = read_features(dtt, dtv, feature_dir)
    dfv = read_features(dtv, dto, feature_dir)
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
    return oos(par, fitpar, dtt, dtv, dto, dte, fit_func=train_linear, metric=metric, feature_groups=feature_groups,
            features=features, verbose=verbose, debug_nfeature=debug_nfeature, do_write_pred=do_write_pred)

def oos_tree(par, fitpar, dtt, dtv, dto, dte, metric='rmse', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None, do_write_pred=True):
    return oos(par, fitpar, dtt, dtv, dto, dte, fit_func=train_tree, metric=metric, feature_groups=feature_groups,
            features=features, verbose=verbose, debug_nfeature=debug_nfeature, do_write_pred=do_write_pred)

def rolling_oos_linear(par, fitpar, st, et, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None):
    '''
    Repeat the linear regression with the sliding windows.

    If the process takes more than half the available memory, and the garbage collector doesn't
    free the data early enough, the job may crash.
    '''
    rolling_oos(par, fitpar, st, et, metric=metric, feature_groups=feature_groups,
        features=features, verbose=verbose, debug_nfeature=debug_nfeature, oos_func=oos_linear)

def rolling_oos_tree(par, fitpar, st, et, metric='r2', feature_groups=None,
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

def plot_target_prediction(dfo, target_name='target', pred_col='pred', nq=100):
    plt.figure(figsize=(12,4))
    ax = plt.subplot(1, 2, 1)
    plot_target_prediction_nquantiles(dfo, target_name, pred_col, nq=nq, ax=ax)
    ax = plt.subplot(1, 2, 2)
    plot_target_prediction_errorbar(dfo, target_name, pred_col, ax=ax)
    plt.tight_layout()

def plot_target_prediction_nquantiles(dfo, target_name='target', pred_col='pred', nq=100, ax=None):
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
    plt.title('target vs prediction (out-of-sample)')
    plt.xlabel('prediciton')
    plt.ylabel('target')
    plt.grid()
    plt.legend()

def plot_target_prediction_errorbar(dfo, target_name='target', pred_col='pred', ax=None):
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

def get_holding_summary(dfpnl, dfo):


    pass

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
