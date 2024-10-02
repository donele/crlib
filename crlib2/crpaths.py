import os

crdata_map = {
    'default': {0: '/mnt/bigdata1/crdata',
            20240701: '/mnt/bigdata2/crdata',
             },
}

def get_idate(dt):
    '''
    Returns the date in yyyymmdd format.
    '''
    idate = int(dt.strftime('%Y%m%d'))
    return idate

def get_data_dir(dt, locale):
    yyyy = dt.year
    mmdd = dt.month * 100 + dt.day
    yyyymmdd = yyyy * 10000 + mmdd

    switchmap = crdata_map[locale] if locale in crdata_map else crdata_map['default']
    switchdate = 0
    for k, v in switchmap.items():
        if k > switchdate and k <= yyyymmdd:
            switchdate = k
    disk = switchmap[switchdate]

    data_dir = f'{disk}/{locale}.{yyyy}/{mmdd:04}'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    return data_dir

def get_feature_dir(par, basedir='/home/jdlee'):
    exch = par['product'][0]
    sym = par['product'][1]
    pname = par['proj_name']
    feature_dir = f'{basedir}/crfeature/{sym}.{exch}.{pname}'
    return feature_dir

def get_fit_dir(par, basedir='/home/jdlee'):
    exch = par['product'][0]
    sym = par['product'][1]
    pname = par['proj_name']
    fit_dir = f'{basedir}/crfit/{sym}.{exch}.{pname}'
    return fit_dir

def get_pred_dir_from_name(par, fit_name):
    pred_dir = f'{get_fit_dir(par)}/{fit_name}/pred'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    return pred_dir

def get_model_dir_from_name(par, fit_name):
    model_dir = f'{get_fit_dir(par)}/{fit_name}/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def get_pred_dir(par, target_name, fit_desc=''):
    fit_name = target_name
    if fit_desc != '':
        fit_name += '.' + fit_desc

    pred_dir = get_pred_dir_from_name(par, fit_name)
    return pred_dir

def get_model_dir(par, fitpar):
    fit_name = fitpar['target_name']
    if 'fit_desc' in fitpar and fitpar['fit_desc'] != '':
        fit_name += '.' + fitpar['fit_desc']

    model_dir = get_model_dir_from_name(par, fit_name)
    return model_dir

def get_pred_path(dt1, dt2, par, fitpar):
    pred_dir = get_pred_dir(par, fitpar['target_name'], fitpar['fit_desc'])
    path = f'{pred_dir}/pred.{get_idate(dt1)}.{get_idate(dt2)}.parquet'
    return path

def get_model_path(dt, par, fitpar):
    model_dir = get_model_dir(par, fitpar)
    path = f'{model_dir}/model.{get_idate(dt)}.{dt.hour:02}{dt.minute:02}.parquet'
    return path

