import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

def linreg_tune(target_name, dft, dfv, metric='r2', feature_groups=None,
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

    #if feature_groups is None:
        #feature_groups=['ret', 'medqimb', 'qimax', 'hilo', 'twret']

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
        model = linreg_tune(target_name, dft, dfv, metric,
                feature_groups=feature_groups,
                verbose=verbose, debug_nfeature=debug_nfeature)
    else: # feature set is fixed.
        model = linreg(target_name, dft, features, verbose)
    return model
