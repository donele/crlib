import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from crlib2 import *

def fit1d():
    parser = argparse.ArgumentParser(prog='fit1d')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target Name')
    parser.add_argument('-a', '--fit-algo', type=str, required=True, help='Fit Algo Name', choices=['lin', 'tree'])
    parser.add_argument('-f', '--fit-window', type=int, required=True, help='Fitting Window')
    parser.add_argument('-v', '--val-window', type=int, required=True, help='Valadation Window')
    parser.add_argument('-o', '--oos-window', type=int, required=True, help='OOS Window')
    parser.add_argument('-y', '--year', type=int, required=True, help='Year')
    parser.add_argument('-m', '--month', type=int, required=True, help='Month')
    parser.add_argument('-d', '--day', type=int, required=True, help='Day')

    a = parser.parse_args()

    par = read_params('params.yaml')
    print(par)
    
    fitpar = {
        'target_name': a.target,
        'fit_desc': f'{a.fit_algo}{a.fit_window}{a.val_window}{a.oos_window}',
        'feature_dir': get_feature_dir(par),
    }
    print(fitpar)
    
    dto = datetime.datetime(a.year, a.month, a.day)
    dtt = dto - timedelta(days = a.val_window + a.fit_window)
    dtv = dto - timedelta(days = a.val_window)
    dte = dto + timedelta(days = a.oos_window)
    
    if a.fit_algo == 'lin':
        oos_linear(par, fitpar, dtt, dtv, dto, dte)
    elif a.fit_algo == 'tree':
        oos_tree(par, fitpar, dtt, dtv, dto, dte)

if __name__ == '__main__':
    fit1d()
