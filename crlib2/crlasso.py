import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

def lasso(target_name, dft, fit_features, alpha=0, verbose=False):
    '''
    Performs a OLS minimization.

    Returns:
        Linear regression coefficients.
    '''
    X = dft.loc[dft.valid, fit_features] # this will be copied inside LinearRegression
    y = dft.loc[dft.valid, target_name]

    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model

def lasso_tune(target_name, dft, dfv, metric='r2', features=None, feature_groups=None,
        verbose=False, debug_nfeature=None, min_data_cnt=10):
    '''
    Select feaures recursively using validation sample.

    Returns:
        Series of selected features and the regression coefficients.
    '''
    best_model = None
    if (dft is None or dfv is None or dft.loc[dft.valid].shape[0] < min_data_cnt
        or dfv.loc[dfv.valid].shape[0] < min_data_cnt):
        return best_model

    if feature_groups is None:
        feature_groups=['ret', 'medqimb', 'qimax', 'hilo', 'twret', 'diff_sum_net_qty']

    allfeatures = [x for x in dft.columns if np.any([x.startswith(g+'_') for g in feature_groups])] if features is None else features

    if not debug_nfeature is None:
        allfeatures = allfeatures[:min(len(allfeatures), debug_nfeature)]

    print(f'Total {len(allfeatures)} features: {allfeatures}')

    best_metric = 0
    for alpha in np.arange(0, 1, 0.1):
        new_metric = 0
        new_model = None
        model = lasso(target_name, dft, allfeatures, alpha=alpha, verbose=verbose)
        if model is None:
            continue
        else:
            Xv = dfv.loc[dfv.valid, allfeatures]
            yv = dfv.loc[dfv.valid, target_name]
            yhatv = pd.Series(model.predict(Xv), index=yv.index)

            # Validation
            r2 = 1 - np.var(yv - yhatv) / np.var(yv)
            rmse = -1e4*(((yv - yhatv)**2).mean()**.5)
            ev = 1e4*(.5*yv[yhatv > yhatv.quantile(0.99)].mean() - .5*yv[yhatv < yhatv.quantile(0.01)].mean())

            # Check if new best is found
            metric_val = r2 if metric == 'r2' else rmse if metric == 'rmse' else ev if metric == 'ev' else 0
            if metric_val > best_metric:
                best_metric = metric_val
                best_model = model

            if verbose:
                print(f'{fit_features} r2={r2:.4f} rmse={rmse:.4f}, ev={ev:.4f}')

            print(f'{metric}={new_metric:.4f}')
            sys.stdout.flush()

    sys.stdout.flush()
    return best_model

def train_lasso(target_name, dft, dfv, metric='r2', feature_groups=None,
        features=None, verbose=False, debug_nfeature=None):
    '''
    Performs a lasso regresison with a fixed feature set, or by using the default
    feature set.

    Returns:
        List of the features and the regression coefficients.
    '''
    model = None
    if features is None: # select features with validation data.
        model = lasso_tune(target_name, dft, dfv, metric,
                feature_groups=feature_groups,
                verbose=verbose, debug_nfeature=debug_nfeature)
    else: # feature set is fixed.
        model = lasso_tune(target_name, dft, dfv, metric,
                features=features,
                verbose=verbose, debug_nfeature=debug_nfeature)
    return model
