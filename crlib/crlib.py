import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import datetime
import yaml
from datetime import timedelta

## Data Hangling

def read_params(path):
    '''
    Reads the project parameters from txt, csv, or yaml file.

    Returns:
        A dictionary.
    '''
    par = None
    if os.path.exists(path):
        ext = path.split('.')[-1]
        if ext in ['txt', 'csv']:
            par = pd.read_csv(path)
            par = {k: par[k][0] for k in par.columns}
            if 'sample_interval' in par:
                par['sample_interval'] = int(par['sample_interval'])
        elif ext == 'yaml':
            with open(path) as f:
                par = yaml.safe_load(f)
    return par

def get_universe():
    '''
    Selects the most liquid product - for now.
    '''
    dfu = pd.read_csv('universe.csv', index_col=[0, 1])
    product = dfu.index[0]
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

def read_data(datatype, dt, data_dir, locale, columns=None, product=None):
    '''
    Reads the tick data of the specified datatype for the time dt.
    '''
    yyyymmdd, hh = parse_dt(dt)
    path = f'{data_dir}/{datatype}.{locale}.{yyyymmdd}-{hh:02d}.parquet.snappy'
    if not os.path.exists(path):
        print(path, ' not found')
        return None

    product_cols = ['exchange', 'symbol']
    readcols = None if columns is None else product_cols + columns
    df = pd.read_parquet(path, columns=readcols)

    if product is None:
        df['date'] = pd.to_datetime(df.ts, unit='us')
        df = df.set_index(product_cols)
    else:
        df = df[(df.exchange==product[0])&(df.symbol==product[1])]
        df['date'] = pd.to_datetime(df.ts, unit='us')
        df = df.set_index('date').sort_index()
    df = df.drop(columns=[x for x in df.columns if columns is not None and x not in columns])

    return df

def read_trade(dt, data_dir, locale, columns=['price', 'abs_qty', 'net_qty', 'ts'], product=None):
    '''
    Reads the trade data for the time dt.
    '''
    df = read_data('trade', dt, data_dir, locale, columns, product)
    return df

def read_bbo(dt, data_dir, locale, columns=['askpx', 'askqty', 'bidpx', 'bidqty', 'adj_askpx', 'adj_bidpx', 'ts'], product=None):
    '''
    Reads the bbo data for the time dt.
    '''
    df = read_data('bbo', dt, data_dir, locale, columns, product)
    return df

def read_midpx(dt, data_dir, locale, columns=['mid_px', 'adj_width', 'adj_mid_px', 'ts'], product=None):
    '''
    Reads the midpx data for the time dt.
    '''
    df = read_data('midpx', dt, data_dir, locale, columns, product)
    return df

def read_range(datatype, product, st, et, data_dir, locale):
    '''
    Reads the tick data of the specified type within the specified time range st and et.
    '''
    if datatype == 'trade':
        func = read_trade
    elif datatype == 'bbo':
        func = read_bbo
    elif datatype == 'midpx':
        func = read_midpx

    dt_range = pd.date_range(st, et, freq='h')
    datalist = []
    for dt in dt_range:
        df0 = func(dt, data_dir, locale, product=product)
        datalist.append(df0)

    if np.all([x is None for x in datalist]):
        return None
    df = pd.concat(datalist)
    return df

def read3(dt1, data_dir, locale, product, func):
    '''
    Reads the data in three consecutive periods from before dt1 to after dt1.
    '''
    dt0 = dt1 - timedelta(hours=1)
    dt2 = dt1 + timedelta(hours=1)

    datalist = []
    datalist.append(func(dt0, data_dir, locale, product=product))
    datalist.append(func(dt1, data_dir, locale, product=product))
    datalist.append(func(dt2, data_dir, locale, product=product))

    if np.any([x is None for x in datalist]):
        return None
    df = pd.concat(datalist)
    return df

def get_features3(dt1, sample_interval, data_dir, locale, product=None, mid_col='price'):
    '''
    Reads the data in three consecutive periods
    '''
    dt0 = dt1 - timedelta(hours=1)
    dt2 = dt1 + timedelta(hours=1)

    dft = read3(dt1, data_dir, locale, product=product, func=read_trade)
    dfb = read3(dt1, data_dir, locale, product=product, func=read_bbo)
    dfm = read3(dt1, data_dir, locale, product=product, func=read_midpx)
    if dft is None or dfb is None or dfm is None:
        return None
    df = get_features(dft, dfb, dfm, sample_interval, mid_col=mid_col)
    df = df.loc[dt1:(dt2-timedelta(microseconds=1))]
    return df

## Plotting

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

def plot_bidask(dft, dfb, dfmid, ifrom='2024-08-15 02:10:16', ito='2024-08-15 02:10:26',
                   adj=False, plot_trade=True):
    '''
    Plots the price series of bbo, trade, and midpx data.
    '''
    fig, ax = plt.subplots(figsize=(12,3))
    fmt = '.-'

    if adj:
        plt.plot(steps(dfb.loc[ifrom:ito, 'adj_bidpx']), fmt, label='adj_bid')
        plt.plot(steps(dfb.loc[ifrom:ito, 'adj_askpx']), fmt, label='adj_ask')
        plt.plot(steps(dfmid.loc[ifrom:ito, 'adj_mid_px']), fmt, label='adj_mid')
    else:
        plt.plot(steps(dfb.loc[ifrom:ito, 'bidpx']), fmt, label='bid')
        plt.plot(steps(dfb.loc[ifrom:ito, 'askpx']), fmt, label='ask')
        plt.plot(steps(dfmid.loc[ifrom:ito, 'mid_px']), fmt, label='mid')
    if plot_trade:
        plt.plot(dft.loc[ifrom:ito].price, '.', label='trade')

    plt.title(f'{ifrom} - {ito}')
    myFmt = mdates.DateFormatter(":%S.%f")
    ax.xaxis.set_major_formatter(myFmt)

    plt.grid()
    plt.legend()

## Features

def future_returns(prc, i):
    r = None
    if prc is not None and len(prc) > i:
        r = (prc.shift(-2**(i-1)) / prc - 1).fillna(0).replace([np.inf, -np.inf], 0)
    return r

def past_returns(prc, i):
    r = None
    if prc is not None and len(prc) > i:
        r = (prc / prc.shift(2**(i-1)) - 1).fillna(0).replace([np.inf, -np.inf], 0)
    return r

def get_returns(df, rangemax, varname, returns_func):
    '''
    Calculates the future returns.
    '''
    serdict = {}
    for i in range(1, rangemax):
        ser = returns_func(df.mid, i)
        ser.name = f'{varname}_{i}'
        serdict[ser.name] = ser

    for i in range(1, rangemax):
        for j in range(i+1, rangemax):
            name1 = f'{varname}_{i}'
            name2 = f'{varname}_{j}'
            ser = serdict[name2] - serdict[name1]
            ser.name = f'diff_{varname}_{i}_{j}'
            serdict[ser.name] = ser

    serlist = list(serdict.values())
    return serlist

def features_future_returns(df, rangemax=10):
    return get_returns(df, rangemax, 'ret', returns_func=future_returns)

def features_past_returns(df, rangemax=10):
    return get_returns(df, rangemax, 'tar', returns_func=past_returns)

def get_features(dft, dfb, dfm, sample_interval, rangemax=8, vtrangemax=14, vmrangemax=8, max_tsince_trade=5, mid_col='price', verbose=False):
    '''
    Calculates the features.

    Returns:
        A dataframe.
    '''

    tgrp = pd.to_datetime(dft.ts // sample_interval * sample_interval, unit='us')
    dfts = dft.groupby(tgrp).agg(
        price=('price', 'last'),
        avg_price=('price', 'mean'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        sum_avg_qty=('abs_qty', 'sum'),
        sum_net_qty=('net_qty', 'sum'),
        last_trade=('price', lambda x: x.index[-1]),
    )
    if verbose:
        print(dfts)

    dfb['qimb'] = ((dfb.askqty - dfb.bidqty) / (dfb.askqty + dfb.bidqty)).fillna(0).replace([np.inf, -np.inf], 0)

    bgrp = pd.to_datetime(dfb.ts // sample_interval * sample_interval, unit='us')
    dfbs = dfb.groupby(bgrp).agg(
        qimb=('qimb', 'last'),
        adj_askpx=('adj_askpx', 'last'),
        adj_bidpx=('adj_bidpx', 'last'),
        max_askqty=('askqty', 'max'),
        max_bidqty=('bidqty', 'max'),
        # med_spreadbp=('spreadbp', 'median'),
    )

    mgrp = pd.to_datetime(dfm.ts // sample_interval * sample_interval, unit='us')
    dfms = dfm.groupby(mgrp).agg(
        adj_width=('adj_width', 'last'),
        adj_mid_px=('adj_mid_px', 'last'),
    )

    # Merge trade and bbo

    allindx = pd.date_range(dfbs.index[0], dfbs.index[-1], freq=datetime.timedelta(microseconds=sample_interval))
    df = pd.concat([dfbs.reindex(allindx), dfts.reindex(allindx), dfms.reindex(allindx)], axis=1)

    # mid price, to be used for some feature calculation.
    # df['mid'] = df['adj_askpx'] - df['adj_bidpx'] # this can be negative!
    df['mid'] = df['adj_mid_px'] # Synthetic price from a model.
    # df['mid'] = df['price'] # Trade price may be a proxy for the mid.

    # ffill
    ffill_cols = ['mid', 'adj_askpx', 'adj_bidpx', 'adj_width', 'last_trade']
    df[ffill_cols] = df[ffill_cols].ffill()

    # tsince_trade
    df['tsince_trade'] = (df.index.to_series().shift(-1) - df.last_trade.fillna(datetime.datetime(1970,1,1))).map(lambda x: x.total_seconds())
    df['valid'] = (df.last_trade.notna()) & (df.tsince_trade <= max_tsince_trade)

    serlst = []

    # BBO related features

    ## Future returns
    serlst.extend(features_future_returns(df))

    ## Past returns
    serlst.extend(features_past_returns(df, rangemax))

    ## Median qimb

    for i in range(1, rangemax+1):
        w = 2**(i-1)
        ser = df.qimb.rolling(window=w, min_periods=1).median().fillna(0).replace([np.inf, -np.inf], 0)
        ser.name = f'medqimb_{i}'
        serlst.append(ser)

    ## qimax

    for i in range(1, rangemax+1):
        w = 2**(i-1)
        aname = f'max_askqty_{i}'
        bname = f'max_bidqty_{i}'
        aser = df.max_askqty.rolling(window=w, min_periods=1).max()
        bser = df.max_bidqty.rolling(window=w, min_periods=1).max()
        aser.name = aname
        bser.name = bname
        qiser = ((aser - bser) / (aser + bser)).fillna(0).replace([np.inf, -np.inf], 0)
        qiser.name = f'qimax_{i}'
        serlst.append(qiser)

    ## Volatility

    volatlist = []
    rser = df.price / df.price.shift(1) - 1
    for i in range(1, vtrangemax+1):
        w = 2**(i+1)
        ser = rser.rolling(window=w, min_periods=1).std()
        ser.name = f'volat_{i}'
        volatlist.append(ser)
    dfvolat = pd.DataFrame(volatlist).T

    for i in range(2, 8):
        for j in range(max(4, i + 1), vtrangemax+1):
            name1 = f'volat_{i}'
            name2 = f'volat_{j}'
            ser = ((dfvolat[name1] - dfvolat[name2]) / dfvolat[name2].abs()).fillna(0).replace([np.inf, -np.inf], 0)
            ser.name = f'diff_volat_{i}_{j}'
            serlst.append(ser)

    # Trade related feature

    ## Hilo

    hllst = []
    for i in range(1, rangemax+1):
        w = 2**(i-1)
        hiser = df.max_price.rolling(window=w, min_periods=1).max()
        loser = df.min_price.rolling(window=w, min_periods=1).min()
        hiser.name = f'max_price_{i}'
        loser.name = f'min_price_{i}'
        hllst.append(hiser)
        hllst.append(loser)
    dfhl = pd.DataFrame(hllst).T
    for i in range(1, rangemax+1):
        hiname = f'max_price_{i}'
        loname = f'min_price_{i}'
        ser = ((df.price - (.5*dfhl[hiname] + .5*dfhl[loname])) / (.5*dfhl[hiname] - .5*dfhl[loname])).fillna(0).replace([np.inf, -np.inf], 0)
        ser.name = f'hilo_{i}'
        serlst.append(ser)


    # Volume related feature

    avglst = []
    netlst = []

    for i in range(1, vmrangemax+1):
        w = 2**(i-1)
        aser = df.sum_avg_qty.rolling(window=w, min_periods=1).sum()
        nser = df.sum_net_qty.rolling(window=w, min_periods=1).sum()
        aser.name = f'sum_avg_qty_{i}'
        nser.name = f'sum_net_qty_{i}'
        avglst.append(aser)
        netlst.append(nser)
    dfavg = pd.DataFrame(avglst).T
    dfnet = pd.DataFrame(netlst).T

    for i in range(1, 5):
        for j in range(i + 1, vmrangemax+1):
            name1 = f'sum_avg_qty_{i}'
            name2 = f'sum_avg_qty_{j}'
            ser = ((dfavg[name1] - dfavg[name2]) / dfavg[name2].abs()).fillna(0).replace([np.inf, -np.inf], 0)
            ser.name = f'diff_sum_avg_qty_{i}_{j}'
            serlst.append(ser)

    for i in range(1, 5):
        for j in range(i + 1, vmrangemax+1):
            name1 = f'sum_net_qty_{i}'
            name2 = f'sum_net_qty_{j}'
            name3 = f'sum_avg_qty_{j}'
            ser = ((dfnet[name1] - dfnet[name2]) / dfavg[name3].abs()).fillna(0).replace([np.inf, -np.inf], 0)
            ser.name = f'diff_sum_net_qty_{i}_{j}'
            serlst.append(ser)

    # Concat all

    df = pd.concat([df] + serlst, axis=1)
    return df

def read_features(dt1, dt2, feature_dir):
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
        df = pd.read_parquet(path)
        if df is not None and df.shape[0] > 0:
            dflist.append(df)
    if len(dflist) > 0:
        df = pd.concat(dflist)
        return df
    return None

## Fitting

def select_features_linear(target_name, dft, dfv, verbose=False, debug_nfeature=None, min_data_cnt=10):
    '''
    Select feaures recursively using validation sample.

    Returns:
        Series of selected features and the regression coefficients.
    '''
    selected_features = []
    best_b = None
    if (dft is None or dfv is None or dft.loc[dft.valid].shape[0] < min_data_cnt
        or dfv.loc[dfv.valid].shape[0] < min_data_cnt):
        return selected_features, best_b

    allfeatures = [x for x in dft.columns if x.startswith('ret_') or x.startswith('medqimb_')
                or x.startswith('qimax_') or x.startswith('hilo')]
    if not debug_nfeature is None:
        allfeatures = allfeatures[:min(len(allfeatures), debug_nfeature)]

    print(f'Total {len(allfeatures)} features: {allfeatures}')

    remaining_features = allfeatures.copy()

    best_r2 = 0
    thres = 1.01    
    while True:
        new_r2 = 0
        new_b = None
        new_feature = None
        for feature in remaining_features:
            # Fit
            fit_features = selected_features + [feature]
            b = linreg(target_name, dft, fit_features, verbose)
            if b is None:
                continue
            else:
                # Validation
                Xv = dfv.loc[dfv.valid, fit_features].copy()
                yv = dfv.loc[dfv.valid, target_name]
                Xv['const'] = 1
                yhatv = Xv @ b
                r2 = 1 - np.var(yv - yhatv) / np.var(yv)

                # Check if new best is found
                if r2 > new_r2 and r2 > best_r2*thres:
                    new_r2 = r2
                    new_b = b
                    new_feature = feature

                if verbose:
                    print(f'{fit_features} r2 {r2:.4f} old_best {best_r2} new_best {new_r2:.4f}')

        if new_feature is None:
            break
        else:
            print(f'Selected {new_feature} r2 {new_r2:.4f}')
            best_r2 = new_r2
            best_b = new_b
            selected_features.append(new_feature)
            remaining_features.remove(new_feature)

    print(f'Selected features: {selected_features}')
    return selected_features, best_b

def linreg(target_name, dft, fit_features, verbose=False):
    '''
    Performs a OLS minimization.

    Returns:
        Linear regression coefficients.
    '''
    X = dft.loc[dft.valid, fit_features].copy()
    y = dft.loc[dft.valid, target_name]
    X['const'] = 1

    # Fit
    b = None
    try:
        b = (np.linalg.inv(X.T @ X) @ X.T @ y).array
    except:
        print('Failed: ', feature)
        if verbose:
            print(dft.shape, X.shape, y.shape)
            print(dft.columns.tolist())
            print(dft.valid.sum())
            print(fit_features)

    return b

def train_linear(target_name, dtt, dtv, dto, feature_dir,
                 features=None, verbose=False, debug_nfeature=None):
    '''
    Performs a linear regression with a fixed feature set, or by selecting
    features recursively.

    Returns:
        List of the selected features and the regression coefficients.
    '''
    dft = read_features(dtt, dtv, feature_dir)
    dfv = read_features(dtv, dto, feature_dir)
    selected_features = None
    best_b = None
    if features is None: # select features with validation data.
        selected_features, best_b = select_features_linear(target_name, dft, dfv, verbose, debug_nfeature)
    else:
        if dfv.shape[0] > 0: # feature set is fixed.
            selected_features = features
            best_b = linreg(target_name, dft, features, verbose)
    return selected_features, best_b

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

def oos_linear(target_name, dtt, dtv, dto, dte, feature_dir, features=None, debug_nfeature=None):
    '''
    Performs fitting with the specified train, validate, and out-of-sample dates.

    Returns:
        Series of predictions.
    '''
    selected_features, best_b = train_linear(target_name, dtt, dtv, dto, feature_dir,
                                             features, verbose=False, debug_nfeature=debug_nfeature)
    if best_b is not None:
        dfo = read_features(dto, dte, feature_dir)
        if dfo is not None and dfo.shape[0] > 0:
            Xo = dfo[selected_features].copy()
            Xo['const'] = 1
            yo = dfo[target_name]
            pred = Xo @ best_b

            dfo = dfo[['adj_askpx', 'adj_bidpx', 'price', 'adj_width', 'adj_mid_px', 'tsince_trade', 'valid', target_name] + selected_features]
            dfo['pred'] = pred

            return dfo
    return None

def pred_linear(target_name, dtt, dtv, dto, dte, feature_dir, features=None, debug_nfeature=None):
    dfo = oos_linear(target_name, dtt, dtv, dto, dte, feature_dir, features, debug_nfeature)
    if dfo is not None and dfo.shape[0] > 0 and 'pred' in dfo.columns:
        return dfo['pred']
    return None

def rolling_pred_linear(target_name, st, et, feature_dir, fit_window, val_window, oos_window,
                    features=None, debug_nfeature=None):
    '''
    Performs fitting with rolling window between st and et.

    Returns:
        Out of sample prediction.
    '''
    pred_list = []
    dtt = st
    dtv, dto, dte = get_dts(st, fit_window, val_window, oos_window)
    while(dte <= et):
        print(dtt, dtv, dto, dte)
        pred = pred_linear(target_name, dtt, dtv, dto, dte, feature_dir, features, debug_nfeature)
        if pred is not None:
            pred_list.append(pred)

        dtt = dtt + timedelta(hours=oos_window)
        dtv, dto, dte = get_dts(dtt, fit_window, val_window, oos_window)
    if len(pred_list) > 0:
        allpred = pd.concat(pred_list)
        return allpred
    return None

def rolling_oos_linear(target_name, st, et, feature_dir, fit_window, val_window, oos_window,
              fitdesc='', features=None, debug_nfeature=None):
    '''
    Performs fitting with rolling window between st and et.

    Returns:
        Out of sample test result with prediction.
    '''
    oospred = rolling_pred_linear(target_name, st, et, feature_dir, fit_window, val_window, oos_window,
                             features, debug_nfeature)
    dfo = read_features(st + timedelta(hours=fit_window+val_window), et, feature_dir)
    if dfo is not None and dfo.shape[0] > 0:
        dfo = dfo[['adj_askpx', 'adj_bidpx', 'price', 'adj_width', 'adj_mid_px', 'tsince_trade', 'valid', target_name]]
        dfo['pred'] = oospred
        oosdir = f'{feature_dir}/fit/{target_name}'
        if not os.path.exists(oosdir):
            os.makedirs(oosdir)
        _fitdesc = fitdesc if len(fitdesc) == 0 else '_'+fitdesc
        path = f'{oosdir}/oos{_fitdesc}_{get_idate(dfo.index[0])}_{get_idate(dfo.index[-1])}.parquet'
        dfo.to_parquet(path)
        print(f'oos pred written to {path}')
        return dfo
    return None

def plot_target_prediction(dfo, target_name):
    '''
    Plots target vs prediciton in 100 prediction quantiles.
    '''
    corr = dfo.pred.corr(dfo[target_name])
    plt.figure()
    qc = pd.qcut(dfo.pred, 100, duplicates='drop')
    tarpred = dfo.groupby(qc)[target_name].mean()
    mean_preds = dfo.groupby(qc).pred.mean()
    plt.plot(mean_preds, tarpred, label=f'corr {corr:.4f}')
    plt.title('target vs prediction (out-of-sample)')
    plt.xlabel('prediciton')
    plt.ylabel('target')
    plt.grid()
    plt.legend()

def plot_target_prediction_errorbar(dfo, target_name):
    '''
    Plots target vs prediction in 13 quantiles of varying sizes.
    '''
    plt.figure()
    bc = pd.cut(dfo.pred, dfo.pred.quantile([0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99, 1]),
               duplicates='drop')
    bcgrp = dfo.groupby(bc)[target_name].agg(['mean', 'std'])
    bcx = dfo.groupby(bc).pred.mean()
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
    return pnl, position

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
        pnl, pos = get_pnl(dfo, target_name, feebp)
        cols.append(pnl)
        cols.append(pos)

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
    feebp_list = dfpnl.columns.get_level_values(0).unique()
    for feebp in feebp_list:
        pnl = dfpnl[feebp]['pnl']
        # d_shrp = get_daily_sharpe(pnl)
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
    # pertrade_list = []
    feebp_list = dfpnl.columns.get_level_values(0).unique()
    for feebp in feebp_list:
        df = dfpnl[feebp]
        pos = df['pos']
        pnl = df['pnl']

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

        dfpnltrd = df.groupby((df.pos.shift() != df.pos).cumsum()).agg(
            pos=('pos', 'mean'), pnl=('pnl', lambda x: x.sum()))
        dfpnltrd = dfpnltrd[dfpnltrd.pos != 0]

        w_rat = len(dfpnltrd[dfpnltrd.pnl > 0]) / len(dfpnltrd)
        bw_rat = len(dfpnltrd[(dfpnltrd.pnl > 0)&(dfpnltrd.pos > 0)]) / len(dfpnltrd[dfpnltrd.pos > 0])
        sw_rat = len(dfpnltrd[(dfpnltrd.pnl > 0)&(dfpnltrd.pos < 0)]) / len(dfpnltrd[dfpnltrd.pos < 0])

        gpt = dfpnltrd.pnl.mean()*1e4
        b_gpt = dfpnltrd[dfpnltrd.pos > 0].pnl.mean()*1e4
        s_gpt = dfpnltrd[dfpnltrd.pos < 0].pnl.mean()*1e4
        (w_mean, w_std) = dfpnltrd[dfpnltrd.pnl>0].pnl.agg(['mean', 'std'])*1e4
        (l_mean, l_std) = dfpnltrd[dfpnltrd.pnl<0].pnl.agg(['mean', 'std'])*1e4

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
            d_pnl = round(d_pnl, 1),
            d_shrp = round(d_shrp, 2),

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
    return ['w_rat', 'bw_rat', 'sw_rat', 'gpt', 'b_gpt', 's_gpt', 'w_mean', 'w_std',
       'l_mean', 'l_std']

def print_markdown(dftrdsumm):
    '''
    Prints out the formatted text to be included in the markdown document as a table.
    '''
    print('|fee|', '|'.join(dftrdsumm.columns), '|')
    print('|---|', '|'.join(['--:'] * len(dftrdsumm.columns)), '|')
    for indx, row in dftrdsumm.iterrows():
        print('|', indx, '|', '|'.join(row.astype(str).tolist()), '|')
