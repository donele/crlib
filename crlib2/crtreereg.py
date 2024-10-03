import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass

@dataclass
class LGBParam:
    use_data_pct: int
    max_depth: int
    num_leaves: int
    min_child_samples: int
    n_estimators: int = 100

def lgbreg(target_name, dft, dfv, features, lgbparam, lgbverbose=False):
    use_data_len = len(dft) * lgbparam.use_data_pct // 100
    X = dft.iloc[-use_data_len:].loc[dft.valid, features]
    y = dft.iloc[-use_data_len:].loc[dft.valid, target_name]
    Xv = dfv.loc[dfv.valid, features]
    yv = dfv.loc[dfv.valid, target_name]

    verbosity = 1 if lgbverbose else -1
    reg_par = {
        'metric': 'rmse',
        'n_estimators': lgbparam.n_estimators,
        'max_depth': lgbparam.max_depth,
        'num_leaves': lgbparam.num_leaves,
        'min_child_samples': lgbparam.min_child_samples,
        'force_row_wise': True,
        'verbosity': verbosity,
    }
    eval_period = 1 if lgbverbose else 0
    fit_par = {
        'eval_set': [(X, y), (Xv, yv)],
        'eval_names': ['train', 'valid'],
        'callbacks': [lgb.early_stopping(stopping_rounds=3, verbose=0), lgb.log_evaluation(period=eval_period)],
    }
    model = lgb.LGBMRegressor(**reg_par)
    model.fit(X, y, **fit_par)
    return model

def get_best_row(df):
    df = df.sort_values(by='va')
    best_row = df.iloc[-1]
    return best_row

def lgbreg_tune(target_name, dft, dfv, metric='rmse', features=None, feature_groups=None,
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
        feature_groups=['diff_ret', 'medqimb', 'qimax', 'hilo', 'twret', 'diff_sum_net_qty']

    allfeatures = [x for x in dft.columns if np.any([x.startswith(g+'_') for g in feature_groups])] if features is None else features

    if not debug_nfeature is None:
        allfeatures = allfeatures[:min(len(allfeatures), debug_nfeature)]

    print(f'Total {len(allfeatures)} features: {allfeatures}')

    n_rep = 18
    n_sim = 40
    selection_early_stop = True

    max_best_va = 0
    udp_range = [10, 100]
    mxd_range = [2, 7]
    nl_range = [10, 100]
    mcs_exp_range = [6, 14]

    final_model = None
    score_list = []
    features = allfeatures.copy()
    for irep in range(n_rep):
        print(f'Starting rep {irep}. udp: {udp_range}, mxd: {mxd_range}, nl: {nl_range}, mcs: {mcs_exp_range}')

        rep_n_sim = n_sim if irep % 2 == 0 else int(n_sim / 2)
        for isim in range(rep_n_sim):
            udp = np.random.randint(*udp_range)
            mxd = np.random.randint(*mxd_range)
            nl = np.random.randint(*nl_range)
            mcs = int(2**np.random.uniform(*mcs_exp_range))

            lgbparam = LGBParam(udp, mxd, nl, mcs)
            model = lgbreg(target_name, dft, dfv, features, lgbparam, verbose)

            Xv = dfv.loc[dfv.valid, features]
            yv = dfv.loc[dfv.valid, target_name]
            valid_score = model.score(Xv, yv)
            print(f'[{isim}:{mxd}.{mcs}]{valid_score:.4f} ', end='')
            sys.stdout.flush()
            dfimportance = pd.DataFrame({'name': model.feature_name_, 'importance': model.feature_importances_})
            score_list.append([udp, mxd, nl, mcs, valid_score, dfimportance, model])
        print(f'\n', end='')

        df0 = pd.DataFrame(data=score_list, columns=['udp', 'mxd', 'nl', 'mcs', 'va', 'imp', 'model'])
        if irep % 2 == 0:
            df1 = df0.loc[df0.va >= df0.va.quantile(0.9)]
            udp_range = [max(5, int(df1.udp.min()) - 10), min(100, int(df1.udp.max()) + 10)]
            mxd_range = [max(2, int(df1.mxd.min()) - 1), int(df1.mxd.max()) + 2]
            nl_range = [max(2, int(df1.nl.min()) - 1), int(df1.nl.max()) + 2]
            mcs_exp_range = [int(np.log2(df1.mcs.min())) - 1, int(np.log2(df1.mcs.max())) + 1]
        else:
            best_row = get_best_row(df0)
            best_va = best_row.va
            print(f'best udp: {best_row.udp}, mxd: {best_row.mxd}, nl: {best_row.nl}, mcs: {best_row.mcs}, va: {best_row.va:.4}\n')

            if best_va > max_best_va:
                max_best_va = best_va;
            elif selection_early_stop:
                break

            model0 = best_row.model
            final_model = model0

            dfimp = best_row.imp
            dfimp = dfimp.sort_values(by='importance')
            features = dfimp.iloc[-int(len(dfimp)*2/3):]['name'].tolist()
            print(f'Reduced {len(features)} features: {features}')
            score_list = []

    return final_model

def train_tree(target_name, dft, dfv, metric='rmse', feature_groups=None,
                 features=None, verbose=False, debug_nfeature=None):
    '''
    Performs a boosted tree regression with a fixed feature set, or by using
    default set of features.

    Returns:
        List of the used features and the regression coefficients.
    '''
    selected_features = None
    model = None
    if features is None: # select features with validation data.
        model = lgbreg_tune(target_name, dft, dfv, metric,
                feature_groups=feature_groups,
                verbose=False, debug_nfeature=debug_nfeature)
    else:
        model = lgbreg_tune(target_name, dft, dfv, metric,
                features=features,
                verbose=False, debug_nfeature=debug_nfeature)
    return model
