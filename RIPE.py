# -*- coding: utf-8 -*-
"""
Created on 22 sept. 2016
@author: VMargot
"""
import copy
import os
import sys
import ast
import warnings
import time
import re
import tarfile
import shutil
import socket
import traceback
import operator

from collections import Counter
from itertools import compress

try:
    import tqdm
except ImportError:
    tqdm = None
    raise ImportError("The tqdm package is required to run this program")

try:
    import cPickle
except ImportError:
    cPickle = None
    raise ImportError("The cPickle package is required to run this program")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    plt = None
    patches = None
    raise ImportError("The matplotlib package is required to run this program")

try:
    import seaborn as sns
except ImportError:
    sns = None
    raise ImportError("The seaborn package is required to run this program")

try:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import accuracy_score, r2_score
except ImportError:
    BaseEstimator = None
    accuracy_score = None
    r2_score = None
    raise ImportError("The sklearn package is required to run this program")

try:
    import numpy as np
except ImportError:
    np = None
    raise ImportError("The numpy package is required to run this program")

try:
    import multiprocessing as mlp
except ImportError:
    mlp = None
    raise ImportError("The multiprocessing package is required to run this program")

try:
    import pandas as pd
except ImportError:
    pd = None
    raise ImportError("The pandas package is required to run this program")

try:
    import scipy.fftpack
    import scipy.stats
    import scipy.cluster.hierarchy as hac
    import scipy.spatial.distance as scipy_dist
    from scipy.stats import t, norm
except:
    raise ImportError("The scipy package is required to run this program")

try:
    import data_utils
except ImportError:
    data_utils = None
    raise ImportError("Don't find the package data_utils")

try:
    import logging
    from logging.handlers import RotatingFileHandler
except ImportError:
    logging = None
    RotatingFileHandler = None
    raise ImportError("The logging package is required to run this program")


# warnings.filterwarnings('error')
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

"""
---------
Functions
---------
"""


def init_logger(path_out, learning_name):
    # création de l'objet logger qui va nous servir à écrire dans les logs
    logger = logging.getLogger(learning_name)
    # on met le niveau du logger à INFO, comme ça il écrit tout sauf les DEBUG
    logger.setLevel(logging.DEBUG)

    # création d'un formateur qui va ajouter le temps, le niveau
    # de chaque message quand on écrira un message dans le log
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    # création d'un handler qui va rediriger une écriture du log vers
    # un fichier en mode 'append', avec 1 backup et une taille max de 100Mo
    file_handler = RotatingFileHandler(path_out + 'Learning.log',
                                       maxBytes=100000000, backupCount=1)
    # on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
    # créé précédement et on ajoute ce handler au logger
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # création d'un second handler qui va rediriger chaque écriture de log
    # sur la console
    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.INFO)
    logger.addHandler(steam_handler)

    return logger


def calc_vars(learning, feature_name, close_var, proc_id, out_q=None):
    map(lambda var: learning.calc_var(var, close_var, out_q), feature_name)
    out_q.put(learning.get_param('bins'))
    out_q.put(proc_id)


def calc_upcp(learning, rules_list, y, ymean, ystd,
              var_horizon, method, n, cov_min, pen,
              close_var, proc_id, out_q):

    path_acceptable = learning.get_param('path_acceptable')
    path_desciption = learning.get_param('path_desciption')
    th = learning.get_param('th')

    for rg in rules_list:
        feature_name = rg.conditions.get_param('features_names')
        xcol = learning.load_var(feature_name)

        if xcol is not None:
            try:
                rg.calc_stats(x=xcol, y=y, ymean=ymean, ystd=ystd,
                              var_horizon=var_horizon, method=method,
                              n=n, cov_min=cov_min, pen=pen,
                              path_acceptable=path_acceptable,
                              path_desciption=path_desciption,
                              close_var=close_var,
                              th=th)
                out_q.put(rg)
            except Exception as e:
                learning.message('Error with rule %s' % str(rg), 2)
                learning.message('Message is: %s' % e.message, 2)
                out_q.put(None)

    out_q.put(proc_id)


def find_upcp(learning, ruleset_candidate, ruleset_cp1, cp, proc_id, out_q):
    if cp == 2:
        for i in range(len(ruleset_candidate)):
            rg = ruleset_candidate[i]
            try:
                rules_list = map(lambda rg_cp1: rg.intersect(rg_cp1, cp, learning),
                                 ruleset_cp1[i+1:])
                out_q.put(rules_list)
            except Exception as e:
                learning.message('Error with rule %s' % str(rg), 2)
                learning.message('Message is: %s' % e.message, 3)
                out_q.put(None)

    else:
        for i in range(len(ruleset_candidate)):
            rg = ruleset_candidate[i]
            try:
                rules_list = map(lambda rg_cp1: rg.intersect(rg_cp1, cp, learning),
                                 ruleset_cp1)
                out_q.put(rules_list)

            except Exception as e:
                learning.message('Error with rule %s' % str(rg), 2)
                learning.message('Message is: %s' % e.message, 3)
                out_q.put(None)

    out_q.put(proc_id)


def calc_clpt(learning, ruleset, proc_id, out_q):
    var_horizon = learning.get_param('varhorizon')
    method = learning.get_param('calcmethod')
    n = learning.get_param('n')
    y = learning.get_param('y')
    ymean = learning.get_param('ymean')
    ystd = learning.get_param('ystd')
    cov_min = learning.get_param('covmin')
    pen = learning.get_param('pen')
    path_acceptable = learning.get_param('path_acceptable')
    path_desciption = learning.get_param('path_desciption')
    th = learning.get_param('th')

    for rg in ruleset:
        try:
            feature_name = rg.conditions.get_param('features_names')
            xcol = learning.load_var(feature_name)
            if xcol is not None:
                rg.calc_stats(x=xcol, y=y, ymean=ymean, ystd=ystd,
                              var_horizon=var_horizon, method=method,
                              n=n, cov_min=cov_min, pen=pen,
                              path_acceptable=path_acceptable,
                              path_desciption=path_desciption,
                              th=th)

                if rg.get_param('out') is False:
                    out_q.put(rg)
                else:
                    out_q.put(None)

        except Exception as e:
            learning.message('Error with rule %s' % str(rg), 2)
            learning.message('Message is: %s' % e.message, 2)
            out_q.put(None)

    out_q.put(proc_id)


def do_pred_mat(learning, rules_list, in_sample, out_q, i):
    for rg in rules_list:
        try:
            pred = rg.get_param('pred')
            pred_vect = pred * rg.calc_activation(learning=learning,
                                                  in_sample=in_sample)
            out_q.put((rg, pred_vect))
        except Exception as e:
            learning.message('Message is: %s' % e.message, 2)
            out_q.put(None)

    out_q.put(i)


def do_clustering(learning, rs, cov_min, cov_max, out_q, sign):

    new_rs = copy.copy(rs)
    block_size = learning.get_param('block_size')
    nb_clusters = learning.get_param('nb_clusters')
    nb_clusters /= 2

    while len(new_rs) > nb_clusters:
        rs = RuleSet([])
        if block_size == 0:
            delta = 0
        else:
            delta = len(new_rs) / block_size

        for i in range(delta + 1):
            if i == delta:
                sub_rs = RuleSet(new_rs[i * block_size:])
            else:
                sub_rs = RuleSet(new_rs[i * block_size:(i + 1) * block_size])

            sub_rs = sub_rs.extract_greater('cov', cov_min)
            sub_rs = sub_rs.extract_least('cov', cov_max)
            sub_rs = sub_rs.extract_from_cluster(learning, nb_clusters)
            rs.extend(sub_rs)

        new_rs = copy.deepcopy(rs)
        del rs

    out_q.put(new_rs)
    out_q.put(sign)


def fast_selection(olearning, ruleset, name='', out_q=None):
    """
    Returns a subset of a given ruleset. This subset is seeking by
    minimization/maximization of the criterion on the training set
    """
    inter_max = olearning.get_param('intermax')
    i = 1
    selected_rs = RuleSet(ruleset[:i])

    pbar = tqdm.tqdm(range(len(selected_rs), len(ruleset) - 1),
                     ncols=60, desc='Selection ' + name)

    new_rule = ruleset[i]
    while new_rule.get_param('crit') > 0:

        utest = map(lambda rg: new_rule.union_test(rg, inter_max, olearning),
                    selected_rs)

        if all(utest) and new_rule.union_test(selected_rs,
                                              inter_max, olearning):
            selected_rs.append(new_rule)

        i += 1
        pbar.update()
        if i <= len(ruleset) - 1:
            new_rule = ruleset[i]
        else:
            break

    pbar.close()
    if out_q is None:
        return selected_rs
    else:
        out_q.put(selected_rs)
        out_q.put(name)


def optimize_selection(olearning, ruleset, name='', out_q=None):
    """
    Returns a subset of a given ruleset. This subset is seeking by
    minimization/maximization of the criterion on the training set
    """
    y = olearning.get_param('y')
    ymean = olearning.get_param('ymean')
    ystd = olearning.get_param('ystd')
    maximized = olearning.get_param('maximized')
    method = olearning.get_param('calcmethod')
    n = olearning.get_param('n')
    pen = olearning.get_param('pen')
    inter_max = olearning.get_param('intermax')
    var_horizon = olearning.get_param('varhorizon')

    nb_rules = 1
    # Then optimization
    selected_rs = RuleSet(ruleset[:nb_rules])
    old_crit = selected_rs.calc_crit(y, ymean,
                                     ystd, var_horizon,
                                     method, n, pen, olearning)[0]
    crit_evo = [old_crit]

    pbar = tqdm.tqdm(range(len(selected_rs), len(ruleset) - 1),
                     ncols=60, desc='Selection ' + name)

    for i in range(len(selected_rs) + 1, len(ruleset)):
        rules_list = [copy.deepcopy(selected_rs)]  # List of ruleset
        crit_list = [old_crit]
        new_rule = ruleset[i]

        if len(rules_list[0]) > 1:
            # Permutation part
            for j in range(len(rules_list[0])):
                selected_rs = copy.deepcopy(rules_list[0])
                selected_rs.pop(j)

                utest = map(lambda rg: new_rule.union_test(rg, inter_max, olearning),
                            selected_rs)

                if all(utest) and new_rule.union_test(selected_rs,
                                                      inter_max, olearning):
                    selected_rs.insert(j, new_rule)
                    rules_list += [copy.deepcopy(selected_rs)]
                    new_crit = rules_list[-1].calc_crit(y, ymean,
                                                        ystd, var_horizon,
                                                        method, n,
                                                        pen, olearning)[0]
                    crit_list += [new_crit]

            # Try to add a new new rule in the selected_rs
            selected_rs = copy.deepcopy(rules_list[0])
            utest = map(lambda rg: new_rule.union_test(rg, inter_max, olearning),
                        selected_rs)

            if all(utest) and new_rule.union_test(selected_rs, inter_max, olearning):
                selected_rs.append(new_rule)
                rules_list += [copy.deepcopy(selected_rs)]
                new_crit = rules_list[-1].calc_crit(y, ymean,
                                                    ystd, var_horizon,
                                                    method, n,
                                                    pen, olearning)[0]

                crit_list += [new_crit]

        else:
            if new_rule.union_test(selected_rs, inter_max, olearning):
                selected_rs.append(new_rule)
                rules_list += [copy.deepcopy(selected_rs)]
                new_crit = rules_list[-1].calc_crit(y, ymean,
                                                    ystd, var_horizon,
                                                    method, n,
                                                    pen, olearning)[0]

                crit_list += [new_crit]

        if maximized:
            ruleset_idx = int(np.argmax(crit_list))
        else:
            ruleset_idx = int(np.argmin(crit_list))

        selected_rs = rules_list[ruleset_idx]

        old_crit = crit_list[ruleset_idx]

        if ruleset_idx != 0:
            crit_evo += [old_crit]

        pbar.update()

    pbar.close()
    if out_q is None:
        return selected_rs
    else:
        out_q.put(selected_rs)
        out_q.put(name)


def dist(u, v):
    u = np.sign(u)
    v = np.sign(v)
    num = np.dot(u, v)
    deno = min(np.dot(u, u),
               np.dot(v, v))

    return 1 - num / deno


def mse_function(y_hat, y):
    """
    Mean Square Error
    mse_function(x,y) = (x-y)^2
    """
    error_vect = y_hat - y
    crit = np.nanmean(pow(error_vect, 2))

    return crit


def mae_function(y_hat, y):
    """
    Mean Absolute Error
    mse_function(x,y) = |x-y|
    """
    error_vect = np.abs(y_hat - y)
    crit = np.nanmean(error_vect)

    return crit


def get_rank(y, nb_picking, reverse=False):
    y_rank = copy.copy(y)
    sorted_val = sorted(y_rank.values, reverse=reverse)
    index = [sorted_val.index(v) for v in y_rank.values]
    temp = pd.Series(index, index=y_rank.index)

    th = temp.groupby(level=0).max() - nb_picking
    temp = temp.unstack()
    temp['th'] = th
    temp = temp.apply(lambda row: row >= row['th'], axis=1)
    del temp['th']

    test = temp.stack().astype('int')

    return test


def get_first_activation(serie):
    temp = serie.loc[serie != 0.0]
    return temp.index.get_level_values(0)[0]


def get_nb_activation(pred_vect, var_horizon):
    bet = pred_vect * 1.0 / var_horizon
    exit_pos = bet.shift(int(var_horizon)).fillna(0.0)
    gain_series = (bet - exit_pos).cumsum()

    gain_series.fillna(0.0, inplace=True)
    gain_series = gain_series.round(5)

    df = pd.DataFrame(index=gain_series.index, columns=['diff'])
    temp = np.diff(gain_series.astype('bool').astype('int'))
    df['diff'].iloc[:-1] = np.roll(temp, -1)
    df.fillna(0.0, inplace=True)

    datename = df.index.names[0]
    if datename is None:
        datename = 'index'
    dates = df.reset_index().groupby('diff')[datename].apply(list)

    if 1 in dates.index:
        start = dates.loc[1]
    else:
        start = list([df.index[0]])
    if -1 in dates.index:
        end = dates.loc[-1]
    else:
        end = list([df.index[-1]])
    if len(start) > len(end):
        end += [df.index[-1]]
    elif len(start) < len(end):
        start += [df.index[0]]

    return len(start)


def gain(pred_vect, y):
    gain_serie = (pred_vect * y).cumsum()
    y_real = y.cumsum()

    return gain_serie, y_real


def gain_multi(pred_vect, y):

    y_real = y.groupby(level=0).mean()

    nb_stock_inactive = y.loc[pred_vect == 0].groupby(level=0).count()
    nb_stock = y.groupby(level=0).count()

    delta_w = (1 + pred_vect.replace(0.0, np.nan)).fillna(0)
    w_inactive = (nb_stock - delta_w.groupby(level=0).sum()) / nb_stock_inactive

    map(lambda ide: delta_w.loc[(ide,)].replace(0.0, w_inactive.loc[ide],
                                                inplace=True),
        w_inactive.index)

    new_perf = (y * delta_w).groupby(level=0).mean()
    gain_sum = new_perf.cumsum()

    return gain_sum, y_real


def pen_gain(pred_vect, x, y, n, var_horizon, pen, close_var=None):
    """
    Values are different from the last version because in this
    one y is not centered before the calculation of the pen_gain
    """
    n = int(n)

    first_date = map(lambda x_col: data_utils.get_first_date(x_col), x)
    first_date = max(first_date)
    pred_vect = pred_vect.loc[pred_vect.index >= first_date]
    y = y.loc[y.index >= first_date]

    if close_var is None:
        bet_vect = pred_vect * 1.0 / var_horizon
        gain_series = bet_vect * y
    else:
        bet_vect = pred_vect * 1.0 / var_horizon
        exit_vect = bet_vect.shift(var_horizon).fillna(0.0)
        cbet = (bet_vect - exit_vect).cumsum()
        gain_series = cbet * close_var.pct_change(1).shift(-1).fillna(0.0)

    nb_activation = get_nb_activation(pred_vect, var_horizon)

    delta = len(gain_series) / n

    gain_list = []

    for i in range(delta):
        sub_vect = gain_series.iloc[i * n:(i + 1) * n]

        sub_crit = sub_vect.sum()
        gain_list.append(sub_crit)

    sub_vect = gain_series.iloc[delta * n:]
    if len(sub_vect) > 100:
        sub_crit = sub_vect.sum()
        gain_list.append(sub_crit)

    gain_list = np.array(gain_list)
    if nb_activation != 0:
        if pen == 'std':
            penalty = np.nanstd(gain_list) / np.sqrt(nb_activation)
        elif pen == 'kurtosis':
            penalty = scipy.stats.kurtosis(gain_list) / np.sqrt(nb_activation)
        elif pen == 'skewness':
            penalty = scipy.stats.skew(gain_list) / np.sqrt(nb_activation)
        else:
            penalty = 0.0

        pengain = np.nanmean(gain_list) - penalty
    else:
        pengain = 0

    gain_max = np.max(gain_list)
    gain_min = np.min(gain_list)

    if any(map(lambda g: g == 0, gain_list)):
        out = True
    elif gain_max < 0.0:
        out = True
    elif abs(gain_min) >= gain_max:
        out = True
    else:
        out = False

    return pengain, gain_min, gain_max, nb_activation, out


def pen_gain_multi(pred_vect, x, y, n, pen):
    gain_list = []
    couv_vect = []

    first_date = map(lambda x_col: data_utils.get_first_date(x_col), x)
    first_date = max(first_date)
    pred_vect = pred_vect.loc[pred_vect.index.get_level_values(0) >= first_date]
    y = y.loc[y.index.get_level_values(0) >= first_date]

    dates_index = pred_vect.index.get_level_values(0).unique()
    delta = int(len(dates_index) / n)

    for i in range(delta):
        sub_dates = dates_index[i * n:(i + 1) * n]
        sub_y = y.ix[sub_dates]
        sub_pred = pred_vect.ix[sub_dates]

        delta_w = copy.copy(sub_pred)

        old_perf = sub_y.groupby(level=0).mean()

        nb_stock_inactive = sub_y.loc[sub_pred == 0].groupby(level=0).count()
        nb_stock = sub_y.groupby(level=0).count()

        delta_w = (1 + delta_w.replace(0.0, np.nan)).fillna(0)
        w_inactive = (nb_stock - delta_w.groupby(level=0).sum()) / nb_stock_inactive

        map(lambda ide: delta_w.loc[(ide,)].replace(0.0, w_inactive.loc[ide],
                                                    inplace=True),
            w_inactive.index)

        new_perf = (sub_y * delta_w).groupby(level=0).mean()
        gain_sum = (new_perf - old_perf).sum()
        gain_list.append(gain_sum)

        couv = float(len(sub_pred.loc[sub_pred != 0])) / len(sub_pred)
        couv_vect.append(couv)

    if len(gain_list) > 0:
        # Mean of gain by period
        gain_mean = np.mean(gain_list)

    else:
        # Mean of gain by period
        gain_mean = 0

    nb_activation = np.sum(np.array(couv_vect).astype(bool))
    nb_activation = max(0.5, float(nb_activation))

    gain_list = np.array(gain_list)
    if pen == 'std':
        penalty = np.nanstd(gain_list) / np.sqrt(nb_activation)
    elif pen == 'kurtosis':
        penalty = scipy.stats.kurtosis(gain_list) / np.sqrt(nb_activation)
    elif pen == 'skewness':
        penalty = scipy.stats.skew(gain_list) / np.sqrt(nb_activation)
    else:
        penalty = 0.0

    pengain = gain_mean - penalty

    gain_max = np.max(gain_list)
    gain_min = np.min(gain_list)

    if any(map(lambda g: g == 0, gain_list)):
        out = True
    elif gain_max < 0.0:
        out = True
    elif abs(gain_min) >= gain_max:
        out = True
    else:
        out = False

    return pengain, gain_min, gain_max, nb_activation, out


def calc_rate(y, pred_vect):
    nb_stock = data_utils.get_nb_assets(y)
    if nb_stock > 1:
        rez = y * pred_vect
        pos_val = rez.groupby(level=0).apply(lambda val: val > 0).sum()
        nb_activation = len(filter(lambda val: val != 0, pred_vect))
    else:
        rez = y * pred_vect
        pos_val = len(filter(lambda val: val > 0, rez))
        nb_activation = len(filter(lambda val: val != 0, pred_vect))

    return float(pos_val) / nb_activation


def loss(x, y):
    return pow(x - y, 2)
    # return np.log(1.0+np.exp(-x*y))


def loss_multi(x, Y):
    old_perf = Y.groupby(level=0).mean()

    nb_stock_inactive = Y.loc[x == 0].groupby(level=0).count()
    nb_stock = Y.groupby(level=0).count()

    delta_w = (1 + x.replace(0.0, np.nan))
    w_inactive = (nb_stock - delta_w.fillna(0).groupby(level=0).sum()) / nb_stock_inactive

    rep = delta_w.reset_index(1).fillna(w_inactive)
    rep = rep.set_index('Ids', append=True)
    rep.index.names = Y.index.names
    rep = rep[rep.columns[0]]

    new_perf = (Y * rep).groupby(level=0).mean()
    gain_sum = (new_perf - old_perf)

    return gain_sum


def wexp_weight(cum_loss, eta):
    return map(lambda r: np.exp(-eta * r), cum_loss)


def transfert(pred):
    return pred


def calc_zscore(active_vect, y, ystd, var_horizon, eps):
    active_vect = (active_vect * y.fillna(0)).astype('bool')
    nb_activation = np.nansum(active_vect)

    deno = np.sqrt(nb_activation * 2)
    num = ystd * np.sqrt(var_horizon+1)
    zscore = num / deno
    return norm.ppf(eps) * zscore


def calc_hoeffding(active_vect, y, th):
    y_max = np.nanmax(y)
    y_min = np.nanmin(y)
    n = sum(active_vect)

    num = (y_max - y_min) * np.sqrt(np.log(2./th))
    deno = np.sqrt(2 * n)
    return num / deno


def calc_bernstein(active_vect, y, th):
    y_max = np.nanmax(y)
    v = np.nansum(y ** 2)
    n = sum(active_vect)

    val1 = y_max * np.log(2./th)
    val2 = 72.0 * v * np.log(2./th)
    return 1./(6. * n) * (val1 + np.sqrt(val1 ** 2 + val2))


def calc_coverage(active_vect, x):
    if type(x) == list:
        index = None
        for i in range(len(x)):
            if index is not None:
                index = x[i].loc[index].dropna().index
            else:
                index = x[i].dropna().index
    else:
        index = x.dropna().index
    temp = pd.Series(np.nan, index=index)
    temp.update(active_vect)
    cov = sum(temp) * 1.0 / len(temp)

    return cov


def calc_subcoverage(active_vect, id_level):
    temp = active_vect.groupby(level=id_level).sum()
    levels = len(set(active_vect.index.levels[id_level]))

    subcoverage = float(len(temp.loc[temp != 0])) / levels

    return subcoverage


def calc_prediction(active_vect, y, ymean):
    y_cond = np.extract(active_vect != 0, y)
    pred = np.nanmean(y_cond)
    pred -= ymean
    return pred


def calc_spectrum(active_vect, nb_stocks):
    """
    Calculation of the Spectrum criterion.
    Ratio of the left and right sides sum (1/4 of the all) of
    the square module Fourier transform and the full sum of
    the square module Fourier transform
    """

    if nb_stocks > 1:
        vect = active_vect.groupby(level=0).mean()
    else:
        vect = active_vect

    vect_len = len(vect)
    active_f = scipy.fftpack.fft(vect)

    sum_freq_val = sum((1.0 / vect_len * np.abs(active_f)) ** 2)

    idx = vect_len / (4 * 2)

    # the first term is just at left
    first_term = (1.0 / vect_len * np.abs(active_f[0])) ** 2
    # the first quart is at left and at right
    first_quart = sum(2.0 * (1.0 / vect_len * np.abs(active_f[1:idx + 1])) ** 2)

    spectrum_crit = first_term + first_quart
    spectrum_crit /= sum_freq_val

    return spectrum_crit


def find_maximal_bend(rs):
    crit_val = map(lambda rg: rg.get_param('crit'), rs)

    f_prime = np.diff(crit_val)
    f_second = np.diff(f_prime)
    nb_pts = len(f_second)

    with np.errstate(divide='ignore'):
        bend = map(lambda x: pow(1 + f_prime[x + 1] ** 2, 3. / 2) / f_second[x],
                   range(nb_pts))

    is_finite = np.isfinite(bend)
    bend = map(lambda c, b: abs(c) if b else 0, bend, is_finite)

    return np.argmax(bend)


def select_from_bend(rs, cov_th, nb_max):
    sub_rs = rs.extract_greater('crit', 0)
    sub_rs = sub_rs.extract_greater('cov', cov_th)

    if len(sub_rs) > 2:
        id_max = find_maximal_bend(sub_rs)
        sub_rs = RuleSet(sub_rs[:id_max])

        if len(sub_rs) > nb_max:
            new_rs = RuleSet([])
            sub_rs_pos = sub_rs.extract_greater('pred', 0)
            sub_rs_neg = sub_rs.extract_least('pred', 0)

            alpha = int((1 - 100.0 / len(sub_rs))*100)
            if len(sub_rs_pos) > 0:
                crit_pos = np.percentile(sub_rs_pos.get_rules_param('crit'), alpha)
                new_rs.extend(sub_rs_pos.extract_greater('crit', crit_pos))
            if len(sub_rs_neg) > 0:
                crit_neg = np.percentile(sub_rs_neg.get_rules_param('crit'), alpha)
                new_rs.extend(sub_rs_neg.extract_greater('crit', crit_neg))

            return new_rs

        else:
            return sub_rs
    else:
        return sub_rs


def make_rs(path):
    """
    To create a ruleset from a csv
    """
    df = pd.read_csv(path, index_col=0)
    nb_rules = len(df)
    rs = RuleSet([])
    for i in range(nb_rules):
        rg = df.copy().iloc[i]
        rg.dropna(axis=0, how='all', inplace=True)
        if 'Var1' in rg.index:
            rule = rule_from_csv(rg)
        elif 'Mean pen_gain' in rg.index:
            rule = rule_from_selected_csv(rg)

        if 'TIME' not in rule.get_vars():
            rs.append(rule)

    return rs


def rule_from_selected_csv(rg):
    features_names = rg['VarName']
    bmin = ast.literal_eval(rg['Min'])
    bmax = ast.literal_eval(rg['Max'])
    xmin = ast.literal_eval(rg['Modalities Min'])
    xmax = ast.literal_eval(rg['Modalities Max'])

    if features_names[0] == '[':
        features_names = ast.literal_eval(features_names)
        features_names = tuple(features_names)
        bmin = tuple(bmin)
        bmax = tuple(bmax)
        xmin = tuple(xmin)
        xmax = tuple(xmax)
    else:
        bmin = int(bmin)
        bmax = int(bmax)
        xmin = int(xmin)
        xmax = int(xmax)

    condition = RuleConditions(features_names, bmin, bmax, xmin, xmax)
    return Rule(condition)


def rule_from_csv(rg):
    cp = len(rg.index) / 3

    if cp == 1:
        rule = make_cp1_rule(rg)
    else:
        rule = make_cpup_rule(rg, cp)

    return rule


def make_cp1_rule(rg):
    features_names = ''
    bmin = 0
    bmax = 0
    xmin = 0
    xmax = 0

    name = rg['Var1']
    features_names += name
    bmin += int(rg['Min1'])
    bmax += int(rg['Max1'])
    xmin += 0

    if 'TIME' in name:
        if 'JMOIS' in name:
            xmax += 30
        elif 'MOIS' in name:
            xmax += 12
        elif 'JSEM' in name:
            xmax += 6
    else:
        val = int(name.split('_M')[-1])
        xmax += val - 1

    condition = RuleConditions(features_names, bmin, bmax, xmin, xmax)
    return Rule(condition)


def make_cpup_rule(rg, cp):
    features_names = []
    bmin = []
    bmax = []
    xmin = []
    xmax = []

    for i in range(1, cp + 1):
        name = rg['Var' + str(i)]
        features_names += [name]
        bmin += [int(rg['Min' + str(i)])]
        bmax += [int(rg['Max' + str(i)])]
        xmin += [0]
        if 'TIME' in name:
            if 'JMOIS' in name:
                xmax += [30]
            elif 'MOIS' in name:
                xmax += [12]
            elif 'JSEM' in name:
                xmax += [6]
        else:
            val = int(name.split('_M')[-1])
            xmax += [val - 1]

    features_names = tuple(features_names)
    bmin = tuple(bmin)
    bmax = tuple(bmax)
    xmin = tuple(xmin)
    xmax = tuple(xmax)

    condition = RuleConditions(features_names, bmin, bmax, xmin, xmax)
    return Rule(condition)


def make_learning(learning_path):
    if os.path.isdir(learning_path):
        dropbox_path = learning_path.split('Dropbox/')[0]
        dropbox_path += 'Dropbox/'

        df = pd.read_csv(learning_path + 'Learning_info.csv')
        df = df.copy().iloc[0]
        df.dropna(axis=0, how='all', inplace=True)

        path_in = df['path_data']
        path_in = path_in.split('Dropbox/')[1]
        path_in = os.path.join(dropbox_path, path_in)

        dtend = df['Training End Date']
        dtend = pd.to_datetime(dtend, dayfirst=True)
        target = df['Asset']
        n = int(df['Gain Period Length (Day)'])

        try:
            path_index = df['path_index']
            path_index = os.path.join(dropbox_path, path_index)
        except KeyError:
            path_index = ''
        try:
            path_acceptable = df['path_acceptable']
        except KeyError:
            path_acceptable = ''
        try:
            path_desciption = df['path_desciption']
        except KeyError:
            path_desciption = ''
        try:
            dtstart = df['Training Start Date']
        except KeyError:
            dtstart = ''
        try:
            reduced = df['reduced']
        except KeyError:
            reduced = False
        try:
            centered = df['centered']
        except KeyError:
            centered = False
        try:
            normalized = df['normalized']
        except KeyError:
            normalized = False

        try:
            varhorizon = df['Var_horizon']
        except KeyError:
            try:
                varhorizon = int(target.split('RCR')[-1])
            except ValueError:
                varhorizon = re.findall(r'\d+', target)
                if len(varhorizon) > 1:
                    print 'Error to find var Horizon'
                    sys.exit()
                else:
                    varhorizon = int(varhorizon[0])
            print 'Var Horizon is %s' % varhorizon

        ymean = 0
        ystd = 1
        method = 'pen_gain'
        covmin = 0.01
        nb_buckets = 10
        logger = init_logger(learning_path, target)

        learning = Learning(logger=logger, path_in=path_in, path_index=path_index,
                            dtend=dtend, dtstart=dtstart, target=target,
                            varhorizon=varhorizon, ystd=ystd, ymean=ymean,
                            method=method, covmin=covmin, n=n, nb_buckets=nb_buckets,
                            normalized=normalized, reduced=reduced, centered=centered,
                            path_acceptable=path_acceptable, path_desciption=path_desciption)

    elif os.path.splitext(learning_path)[1] == '.gz':
        learning = load_learning(learning_path)

    else:
        print 'Error with the path %s' % learning_path
        sys.exit()

    return learning


def update_rs(rs, learning):
    target = learning.get_param('target')
    y = learning.load_target(target)
    var_horizon = learning.get_param('varhorizon')
    method = learning.get_param('calcmethod')
    n = learning.get_param('n')
    path_acceptable = learning.get_param('path_acceptable')
    path_desciption = learning.get_param('path_desciption')
    th = learning.get_param('th')

    if learning.get_param('normalized'):
        ymean = 0
        ystd = 1
    elif learning.get_param('centered'):
        ystd = y.std()
        ymean = 0
    elif learning.get_param('reduced'):
        ymean = y.mean()
        ystd = 1
    else:
        ymean = y.mean()
        ystd = y.std()

    learning.set_params(ymean=ymean)
    learning.set_params(ystd=ystd)

    cov_min = learning.get_param('covmin')
    pen = learning.get_param('pen')

    updated_rs = RuleSet([])
    for rule in rs:
        condi_dict = rule.conditions.__dict__
        for att_name, att_val in condi_dict.iteritems():
            if att_name != 'values':
                if rule.get_param('cp') == 1:
                    condi_dict[att_name] = [att_val]
                else:
                    condi_dict[att_name] = list(att_val)

        vars_name = rule.get_vars()
        x = learning.load_var(vars_name)
        if x is not None:
            rule.calc_stats(x=x, y=y, ymean=ymean, ystd=ystd,
                            var_horizon=var_horizon, method=method,
                            n=n, cov_min=cov_min, pen=pen,
                            save_activation=True,
                            path_acceptable=path_acceptable,
                            path_desciption=path_desciption,
                            th=th)
            # rule.score(X,y)
            updated_rs.append(rule)

    return updated_rs


def load_learning(path, pathx=None, pathy=None):

    if os.path.isfile(path):
        racine = path.split('Dropbox/')[0]
        racine += 'Dropbox/'

        if os.name == 'nt':
            tar_path = path
            tar = tarfile.open(tar_path, "r:gz")
            tar.extractall()
            tar.close()
            pickle_path = path.replace('.tar.gz', '')
            pickle_path_local = pickle_path.split('/')[-1]
            shutil.move(pickle_path_local, pickle_path)
            rep = 0
        else:
            tar_path = path.replace(' ', '\\ ')
            dir_path = os.path.dirname(tar_path)
            rep = os.system('tar -xzf %s -C %s' % (tar_path, dir_path))

        if rep == 0:
            # loading from the pickle
            pickle_path = path.replace('.tar.gz', '')
            learning = cPickle.load(open(pickle_path, "rb"))

            # Deletion of the pickle file
            pickle_path = pickle_path.replace(' ', '\\ ')
            os.system('rm %s' % pickle_path)

            if pathx is None:
                pathx = learning.get_param('pathx')
            if pathy is None:
                pathy = learning.get_param('pathy')

            if os.path.isfile(pathx) is False and os.path.isdir(pathx) is False:
                pathx = os.path.join(racine, pathx.split('Dropbox/')[1])
                pathy = os.path.join(racine, pathy.split('Dropbox/')[1])

            # pathx= pathx.replace('X/', 'Archive derivation 13 07 17/X/')
            # pathy = pathy.replace('Y/', 'Archive derivation 13 07 17/Y/')

            learning.set_params(pathx=pathx)
            learning.set_params(pathy=pathy)

            return learning
        else:
            print 'Error to decompress'
            sys.exit()
    else:
        print 'No tar.gz file here %s' % path
        sys.exit()


def get_variables_count(rs):
    col_vars = map(lambda rg: rg.conditions.get_param('features_names'), rs)
    vars_list = reduce(operator.add, col_vars)
    count = Counter(vars_list)

    return count


class RuleConditions(object):
    """
    Class for binary rule condition

    Warning: this class should not be used directly.
    """
    def __init__(self, features_names,
                 bmin, bmax, xmin, xmax,
                 values=list()):

        self.features_names = features_names

        assert bmin <= bmax, 'Bmin must be smaller or equal than bmax (%s)' \
                             % features_names
        self.bmin = bmin
        self.bmax = bmax
        self.xmin = xmin
        self.xmax = xmax
        self.values = [values]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        features = self.features_names
        return "Var: %s, Bmin: %s, Bmax: %s" % (features, self.bmin, self.bmax)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        to_hash = [(self.features_names[i], self.bmin[i], self.bmax[i])
                   for i in range(len(self.features_names))]
        to_hash = tuple(to_hash)
        return hash(to_hash)

    def transform(self, x_cols):
        """
        Transform dataset.
        Parameters
        ----------
        x_cols: array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        cp = len(self.features_names)
        geq_min = True
        leq_min = True
        not_nan = True
        for i in range(cp):
            geq_min &= np.greater_equal(x_cols[i].fillna(self.bmin[i] - 1),
                                        self.bmin[i])
            leq_min &= np.less_equal(x_cols[i].fillna(self.bmax[i] + 1),
                                     self.bmax[i])
            not_nan &= np.isfinite(x_cols[i])

        res = 1 * (geq_min & leq_min & not_nan)

        return res

    def get_param(self, param):
        assert type(param) == str, 'Must be a string'

        return getattr(self, param)

    def get_attr(self):
        return [self.features_names,
                self.bmin, self.bmax,
                self.xmin, self.xmax]

    """------   Setters   -----"""
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class Rule(object):
    """
    Class for a rule with a binary rule condition
    """
    def __init__(self,
                 rule_conditions):

        assert rule_conditions.__class__ == RuleConditions, \
            'Must be a RuleCondition object'

        self.conditions = rule_conditions
        self._cp = len(rule_conditions.get_param('features_names'))

    def __eq__(self, other):
        return self.conditions == other.conditions

    def __gt__(self, val):
        return self.get_param('pred') > val

    def __lt__(self, val):
        return self.get_param('pred') < val

    def __ge__(self, val):
        return self.get_param('pred') >= val

    def __le__(self, val):
        return self.get_param('pred') <= val

    def __str__(self):
        return 'Rule: ' + self.conditions.__str__()

    def __hash__(self):
        return hash(self.conditions)
    #
    # def test_included(self, rule, learning):
    #     """
    #     Test to know if a rule (self) and an other (other)
    #     are included
    #     """
    #
    #     if (all(intersection == activation_self) or
    #             all(intersection == activation_other)):
    #         return True, None
    #     else:
    #         return False, intersection

    def test_included(self, rule, learning):
        """
        Test to know if a rule (self) and an other (other)
        are included
        """
        activation_self = self.calc_activation(learning=learning)
        activation_other = rule.calc_activation(learning=learning)

        intersection = pd.Series(activation_self * activation_other)

        if (np.allclose(intersection, activation_self, equal_nan=True) or
                np.allclose(intersection, activation_other, equal_nan=True)):
            return None
        else:
            return intersection

    def test_variables(self, rule):
        """
        Test to know if a rule (self) and an other (other)
        have conditions on the same features.
        """
        c1 = self.conditions
        c2 = rule.conditions

        c1_name = c1.get_param('features_names')
        c2_name = c2.get_param('features_names')
        if self.get_param('cp') == 1 and rule.get_param('cp') == 1:
            return c1_name == c2_name

        elif self.get_param('cp') > 1 and rule.get_param('cp') == 1:
            return any(map(lambda var: c2_name == var,
                           c1_name))

        elif self.get_param('cp') == 1 and rule.get_param('cp') > 1:
            return any(map(lambda var: c1_name == var,
                           c2_name))

        else:
            if len(set(c1_name).intersection(c2_name)) != 0:
                return False
            else:
                return True

    def test_cp(self, rule, cp):
        """
        Test to know if a rule (self) and an other (other)
        could be intersected to have a new rule of complexity cp.
        """
        return self.get_param('cp') + rule.get_param('cp') == cp

    # def intersect_test(self, rule, cp, learning):
    #     """
    #     Test to know if a rule (self) and an other (other)
    #     could be intersected.
    #     """
    #     inter_vect = self.test_included(rule=rule, learning=learning)
    #     if inter_vect is not None:
    #         same_variables = self.test_variables(rule)
    #         if same_variables is False:
    #             return self.test_cp(rule, cp), inter_vect
    #         else:
    #             return None
    #     else:
    #         return None

    def intersect_act(self, rule, cp, learning):
        """
        Test to know if a rule (self) and an other (other)
        could be intersected.
        """
        if self.test_cp(rule, cp):
            if self.test_variables(rule) is False:
                return self.test_included(rule=rule, learning=learning)
            else:
                return None
        else:
            return None

    def union_test(self, rule, inter_max=0.80, learning=None):
        """
        Test to know if a rule (self) and an activation vector have
        at more inter_max percent of points in common
        """
        rule_vect = rule.calc_activation(learning=learning)
        self_vect = self.calc_activation(learning=learning)

        pts_inter = np.dot(self_vect, rule_vect)
        pts_vect = np.nansum(rule_vect)
        pts_rule = np.nansum(self_vect)

        ans = ((pts_inter < inter_max * pts_rule) and
               (pts_inter < inter_max * pts_vect))

        return ans

    def intersect_conditions(self, rule):
        conditions_1 = self.conditions
        conditions_2 = rule.conditions

        conditions = map(lambda c1, c2: c1 + c2, conditions_1.get_attr(),
                         conditions_2.get_attr())

        return conditions

    def intersect(self, rule, cp, learning=None):
        if self.get_param('pred') * rule.get_param('pred') > 0:

            intersect_act = self.intersect_act(rule, cp, learning)
            if intersect_act is not None:
                conditions_list = self.intersect_conditions(rule)

                new_conditions = RuleConditions(features_names=conditions_list[0],
                                                bmin=conditions_list[1],
                                                bmax=conditions_list[2],
                                                xmax=conditions_list[4],
                                                xmin=conditions_list[3])
                new_rule = Rule(new_conditions)
                new_rule.set_params(activation=intersect_act)
                return new_rule
            else:
                return None
        else:
            return None

    def complement(self):
        conditions = self.conditions
        values = conditions.values[0].tolist()

        if type(conditions.bmin[0]) not in [str, np.string_]:
            id_max = values.index(conditions.bmax[0])
            id_min = values.index(conditions.bmin[0])

            if (conditions.bmin[0] == conditions.xmin[0] and
                    conditions.bmax[0] == conditions.xmax[0]):
                return [None]

            elif conditions.bmin == conditions.xmin:

                new_conditions = RuleConditions(features_names=conditions.features_names,
                                                bmin=[values[id_max + 1]],
                                                bmax=conditions.xmax,
                                                xmax=conditions.xmax,
                                                xmin=conditions.xmin)

                complement_rule = Rule(new_conditions)
                return [complement_rule]

            elif conditions.bmax == conditions.xmax:
                new_conditions = RuleConditions(features_names=conditions.features_names,
                                                bmin=conditions.xmin,
                                                bmax=[values[id_min - 1]],
                                                xmax=conditions.xmax,
                                                xmin=conditions.xmin)
                complement_rule = Rule(new_conditions)
                return [complement_rule]

            else:
                new_conditions_1 = RuleConditions(features_names=conditions.features_names,
                                                  bmin=conditions.xmin,
                                                  bmax=[values[id_min - 1]],
                                                  xmax=conditions.xmax,
                                                  xmin=conditions.xmin)

                new_conditions_2 = RuleConditions(features_names=conditions.features_names,
                                                  bmin=[values[id_max + 1]],
                                                  bmax=conditions.xmax,
                                                  xmax=conditions.xmax,
                                                  xmin=conditions.xmin)

                complement_rule_1 = Rule(new_conditions_1)
                complement_rule_2 = Rule(new_conditions_2)
                return [complement_rule_1, complement_rule_2]
        else:
            return [None]

    def is_acceptable(self, path_acceptable):
        acceptable = True
        if os.path.isfile(path_acceptable):
            dir_path = os.path.dirname(path_acceptable) + '/'
            sys.path.append(dir_path)

            py_file = path_acceptable.split(dir_path)[-1]

            if 'v2' in py_file:
                try:
                    import acceptablev2 as acc
                except ImportError:
                    acc = None
            else:
                try:
                    import acceptable as acc
                except ImportError:
                    acc = None

            if acc is None:
                print 'Error to import acceptable function'
                sys.exit()
            else:
                acceptable = acc.is_acceptable(self)

            return acceptable
        else:
            return acceptable

    def get_description(self, path_description):
        if os.path.isfile(path_description):
            dir_path = os.path.dirname(path_description) + '/'
            sys.path.append(dir_path)

            py_file = path_description.split(dir_path)[-1]

            if 'v2' in py_file:
                try:
                    import descriptionv2
                    desc = descriptionv2.rule_description(self)
                    self.set_params(description=desc)
                except ImportError:
                    print 'Error to import description function'
                    descriptionv2 = ''
                    self.set_params(description=descriptionv2)
            else:
                try:
                    import description
                    desc = description.description(self)
                    self.set_params(description=desc)
                except ImportError:
                    print 'Error to import description function'
                    description = ''
                    self.set_params(description=description)
        else:
            self.set_params(description='')

    def calc_stats(self, x, y, ymean, ystd, var_horizon=1,
                   method='mse_function', n=252, cov_min=0.01, pen='std',
                   path_acceptable='', path_desciption='', th=0.05,
                   sini_crit='zscore', save_activation=False, close_var=None):

        self.set_params(out=False)
        if hasattr(self, '_activation'):
            active_vect = self.get_param('activation')
        else:
            active_vect = self.calc_activation(x=x)

        nb_stocks = data_utils.get_nb_assets(y)

        if save_activation:
            self.set_params(activation=active_vect)
        elif nb_stocks == 1:
            self.set_params(activation=active_vect)
        else:
            if hasattr(self, '_activation'):
                del self._activation

        cov = calc_coverage(active_vect, x)
        self.set_params(cov=cov)

        if nb_stocks > 1:
            cov_time = calc_subcoverage(active_vect, id_level=0)
            cov_stock = calc_subcoverage(active_vect, id_level=1)
            self.set_params(cov_time=cov_time)
            self.set_params(cov_stock=cov_stock)
        else:
            self.set_params(cov_time=np.nan)
            self.set_params(cov_stock=np.nan)

        if cov < cov_min:
            self.set_params(out=True)
            self.set_params(reason='Rule %s has coverage too small: %s'
                                   % (str(self), str(cov)))
            return

        else:
            pred = calc_prediction(active_vect, y, ymean)
            self.set_params(pred=pred)

            if sini_crit == 'zscore':
                eps = 1 - th / 2.0
                sini_val = calc_zscore(active_vect, y, ystd, var_horizon, eps)
                if sini_val != 0:
                    zscore = abs(pred) * (norm.ppf(eps)/sini_val)
                else:
                    nb_activation = np.nansum(active_vect)
                    num = np.sqrt(nb_activation * 2)

                    first_date = map(lambda x_col: data_utils.get_first_date(x_col), x)
                    first_date = max(first_date)
                    sub_y = y.loc[y.index.get_level_values(0) >= first_date]
                    deno = np.nanstd(sub_y) * np.sqrt(var_horizon + 1)

                    zscore = abs(pred) * num/deno

                self.set_params(zscore=zscore)

            elif sini_crit == 'hoeffding':
                sini_val = calc_hoeffding(active_vect, y, th)

            elif sini_crit == 'bernstein':
                sini_val = calc_bernstein(active_vect, y, th)

            else:
                sini_val = 0

            self.set_params(th=sini_val)

            if abs(pred) < sini_val:
                self.set_params(out=True)
                self.set_params(reason='Rule %s not significant' % (str(self)))
                return

            if self.is_acceptable(path_acceptable) is False:
                self.set_params(out=True)
                self.set_params(reason='Rule %s not acceptable' % (str(self)))
                return

            rez = self.calc_crit(active_vect, x, y, ymean,
                                 var_horizon, method,
                                 n, pen, close_var)

            self.set_params(crit=rez[0])
            self.set_params(crit_min=rez[1])
            self.set_params(crit_max=rez[2])
            self.set_params(nb_activation=rez[3])

            if rez[4] is True:
                self.set_params(out=rez[4])
                self.set_params(reason='Criterium')
                return

            self.get_description(path_desciption)

    def calc_activation(self, x=None, learning=None,
                        index=None, in_sample=False):

        activation = pd.Series()
        if in_sample:
            if hasattr(self, '_activation') and self.get_param('activation') is not None:
                activation = self.get_param('activation')
            elif learning is not None:
                feature_name = self.conditions.get_param('features_names')
                x = learning.load_var(feature_name, index)
                activation = self.conditions.transform(x)
                self.set_params(activation=activation)

        else:
            if x is not None:
                activation = self.conditions.transform(x)

            elif learning is not None:
                feature_name = self.conditions.get_param('features_names')
                x = learning.load_var(feature_name, index)
                activation = self.conditions.transform(x)

        return activation

    def calc_crit(self, active_vect, x, y, ymean, var_horizon,
                  method='mse_function', n=252, pen='std', close_var=None):

        pred = self.get_param('pred')
        pred_vect = active_vect * transfert(pred)
        y -= ymean
        y_fillna = y.fillna(0.0)

        if method == 'mse_function':
            y_fillna = np.extract(pred_vect != 0, y_fillna)
            pred_vect = np.extract(pred_vect != 0, pred_vect)

            crit = mse_function(pred_vect, y_fillna)
            crit_min = np.nan
            crit_max = np.nan
            nb_activation = np.nan
            out = False

            tuple_rez = (crit, crit_min, crit_max, nb_activation, out)

        elif method == 'mae_function':
            y_fillna = np.extract(pred_vect != 0, y_fillna)
            pred_vect = np.extract(pred_vect != 0, pred_vect)

            crit = mae_function(pred_vect, y_fillna)
            crit_min = np.nan
            crit_max = np.nan
            nb_activation = np.nan
            out = False

            tuple_rez = (crit, crit_min, crit_max, nb_activation, out)

        elif method == 'pengain':
            if data_utils.get_nb_assets(y) == 1:
                tuple_rez = pen_gain(pred_vect, x, y_fillna, n,
                                     var_horizon, pen, close_var)
            else:
                tuple_rez = pen_gain_multi(pred_vect, x, y_fillna,
                                           n, pen)
        else:
            raise 'Method %s unknown' % method

        return tuple_rez

    def predict(self, x=None, learning=None, index=None):
        pred = self.get_param('pred')
        activation = self.calc_activation(x=x,
                                          learning=learning,
                                          index=index)

        return pred * activation

    def score(self, x, y, sample_weight=None, score_type='Rate'):
        """
        Returns the coefficient of determination R^2 of the prediction
        if y is continuous. Else if y in {0,1} then Returns the mean
        accuracy on the given test data and labels {0,1}.

        Parameters
        ----------
        x : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        score_type : string-type

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y in R.

            or

        score : float
            Mean accuracy of self.predict(X) wrt. y in {0,1}
        """

        pred_vect = self.predict(x)

        if score_type == 'Rate':
            return calc_rate(y, pred_vect)

        else:
            nan_val = np.argwhere(np.isnan(y))
        if len(nan_val) > 0:
            new_index = y.dropna(inplace=True).index
            pred_vect = pred_vect.loc[new_index]
        if score_type == 'Classification':
            th_val = (min(y) + max(y)) / 2.0
            pred_vect = np.array(map(lambda p: min(y) if p < th_val else max(y),
                                     pred_vect))
            return accuracy_score(y, pred_vect)

        elif score_type == 'Regression':
            return r2_score(y, pred_vect, sample_weight=sample_weight,
                            multioutput='variance_weighted')

    def make_name(self, num, learning=None):
        name = 'R_' + str(num)
        cp = self.get_param('cp')
        name += '(' + str(cp) + ')'
        pred = self.get_param('pred')
        if pred > 0:
            name += '+'
        elif pred < 0:
            name += '-'

        if learning is not None:
            dtstart = learning.get_param('dtstart')
            dtend = learning.get_param('dtend')
            if dtstart is not None:
                name += str(dtstart) + ' '
            if dtend is not None:
                name += str(dtend)
        self.set_params(name=name)

    """------   Getters   -----"""
    def get_param(self, param):
        assert type(param) == str, 'Must be a string'
        assert hasattr(self, '_' + param), \
            'self._%s must be calculate before' % param
        return getattr(self, '_' + param)

    def get_vars(self):
        return self.conditions.get_param('features_names')

    """------   Setters   -----"""
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, '_' + parameter, value)
        return self


class RuleSet(object):
    """
    Class for a ruleset. It's a kind of list of Rule object
    """
    def __init__(self, param):
        if type(param) == list:
            self._rules = param
        elif type(param) == RuleSet:
            self._rules = param.get_rules()

    def __str__(self):
        return 'ruleset: %s rules' % str(len(self._rules))

    def __gt__(self, val):
        return map(lambda rg: rg > val, self._rules)

    def __lt__(self, val):
        return map(lambda rg: rg < val, self._rules)

    def __ge__(self, val):
        return map(lambda rg: rg >= val, self._rules)

    def __le__(self, val):
        return map(lambda rg: rg <= val, self._rules)

    def __add__(self, ruleset):
        return self.extend(ruleset)

    def __getitem__(self, i):
        return self.get_rules()[i]

    def __len__(self):
        return len(self.get_rules())

    def __del__(self):
        if len(self) > 0:
            nb_rules = len(self)
            i = 0
            while i < nb_rules:
                del self[0]
                i += 1

    def __delitem__(self, rg_id):
        del self._rules[rg_id]

    def append(self, rule):
        assert rule.__class__ == Rule, 'Must be a Rule object (try extend)'
        self._rules.append(rule)

    def extend(self, ruleset):
        assert ruleset.__class__ == RuleSet, 'Must be a ruleset object'
        # assert self._oAdLearn == ruleset._oAdLearn , \
        'ruleset must have the same Learning object'
        rules_list = ruleset.get_rules()
        self._rules.extend(rules_list)
        return self

    def insert(self, idx, rule):
        assert rule.__class__ == Rule, 'Must be a Rule object'
        self._rules.insert(idx, rule)

    def pop(self, idx):
        self._rules.pop(idx)

    def extract_greater(self, param, val):
        rules_list = filter(lambda rg: rg.get_param(param) > val, self)
        return RuleSet(rules_list)

    def extract_least(self, param, val):
        rules_list = filter(lambda rg: rg.get_param(param) < val, self)
        return RuleSet(rules_list)

    def extract_cp(self, cp):
        rules_list = filter(lambda rg: rg.get_param('cp') == cp, self)
        return RuleSet(rules_list)

    def extract(self, param, val):
        rules_list = filter(lambda rg: rg.get_param(param) == val, self)
        return RuleSet(rules_list)

    def index(self, rule):
        assert rule.__class__ == Rule, 'Must be a Rule object'
        self.get_rules().index(rule)

    def replace(self, idx, rule):
        self._rules.pop(idx)
        self._rules.insert(idx, rule)

    def sort_by(self, crit, maximized):
        self._rules.sort(key=lambda x: x.get_param(crit),
                         reverse=maximized)

    def to_df(self):
        """
        To transform a ruleset into a pandas DataFrame
        """
        df = pd.DataFrame(index=self.get_rules_name(),
                          columns=['Features_Names', 'BMin', 'BMax',
                                   'Cov', 'Cov_Time', 'Cov_Stock',
                                   'Pred', 'Crit', 'Crit_Min',
                                   'Crit_Max', 'Zscore', 'Th',
                                   'Nb_Activation', 'Cluster', 'Description'])

        for col_name in df.columns:
            att_name = col_name.lower()
            if all(map(lambda rg: hasattr(rg, '_' + att_name),
                       self)):
                df[col_name] = map(lambda rg:
                                   rg.get_param(att_name),
                                   self)

            elif all(map(lambda rg: hasattr(rg.conditions, att_name.lower()),
                         self)):
                df[col_name] = map(lambda rg:
                                   rg.conditions.get_param(att_name),
                                   self)

        df.rename(columns={'Th': 'Threshold'}, inplace=True)
        return df.dropna(axis=1, how='all')

    def save_rules_df(self, path):
        """
        To save a ruleset as a csv
        """
        nb_rules = len(self)
        rep = pd.DataFrame('', index=self.get_rules_name(),
                           columns=list(('RuleId',
                                         'Var1', 'Min1', 'Max1',
                                         'Var2', 'Min2', 'Max2',
                                         'Var3', 'Min3', 'Max3',
                                         'Var4', 'Min4', 'Max4'))
                           )

        for i in range(nb_rules):
            var = self[i].conditions.get_param('features_names')
            bmin = self[i].conditions.get_param('bmin')
            bmax = self[i].conditions.get_param('bmax')

            cp = len(var)
            for j in range(cp):
                var_name = 'Var' + str(j + 1)
                min_name = 'Min' + str(j + 1)
                max_name = 'Max' + str(j + 1)

                rep.loc[i, var_name] = var[j]
                rep.loc[i, min_name] = bmin[j]
                rep.loc[i, max_name] = bmax[j]

        # rep['RuleId'] = map(lambda rule_id: 'Rules_' + str(rule_id), range(nb_rules))
        # rep.set_index('RuleId', inplace=True)
        rep.to_csv(path)

    def calc_weight(self, learning, in_sample):
        """
        Computes the weights for an equally weighted aggregation
        """
        sum_vect = sum(map(lambda rg: rg.calc_activation(learning=learning,
                                                         in_sample=in_sample),
                           self))

        if len(self) > 1:
            weight_vect = map(lambda act: 1. / act if act != 0 else 0, sum_vect)
        else:
            weight_vect = sum_vect

        return weight_vect

    def calc_weight_equi(self, x=None, learning=None, index=None):
        """
        Computes the weights for an equally weighted aggregation
        """
        sum_vect = sum(map(lambda rg: rg.calc_activation(x, learning, index),
                           self))

        if len(self) > 1:
            weight_vect = map(lambda act: 1. / act if act != 0 else 0, sum_vect)
        else:
            weight_vect = sum_vect

        return weight_vect

    def calc_weight_wexp(self, y_app, learning):
        """
        Computes the weights for an exponential aggregation
        """
        activations_vect = map(lambda rg: (rg.get_param('pred') *
                                           rg.calc_activation(learning=learning)),
                               self)

        rg_names = ['Rules_%s' % str(i) for i in range(len(self))]
        activation_mat = pd.DataFrame(activations_vect).T
        activation_mat.columns = rg_names

        is_multi_index = isinstance(activation_mat.index, pd.core.index.MultiIndex)

        if is_multi_index:
            loss_mat = activation_mat.apply(loss_multi, args=(y_app.fillna(0.0),))
        else:
            loss_mat = activation_mat.apply(loss, args=(y_app.fillna(0.0),))

        loss_mat.fillna(0.0, inplace=True)

        loss_max = np.max(loss_mat.values)
        if is_multi_index:
            h_agg = float(len(set(y_app.index.get_level_values(0))))
        else:
            h_agg = float(len(y_app))

        nb_rules = len(self)
        eta = 1. / loss_max * np.sqrt(8. * np.log(nb_rules) / h_agg)

        cumloss_mat = loss_mat.cumsum()
        pi_mat = cumloss_mat.apply(wexp_weight, args=(eta,))

        pi_mat = pi_mat.shift(1).fillna(0.0)
        pi_mat.fillna(1. / len(self))

        pi_mat = pi_mat.div(pi_mat.sum(axis=1), axis=0)

        # online to batch
        weight_vect = pi_mat.mean(axis=0).values
        weight_vect /= sum(weight_vect)

        return weight_vect

    def calc_weight_boa(self, y_app, learning):
        """
        Computes the weights for a BOA aggregation
        """
        activations_vect = map(lambda rg: (rg.get_param('pred') *
                                           rg.calc_activation(learning=learning)),
                               self)

        rg_names = ['Rules_%s' % str(i) for i in range(len(self))]
        activation_mat = pd.DataFrame(activations_vect).T
        activation_mat.columns = rg_names

        is_multi_index = isinstance(activation_mat.index, pd.core.index.MultiIndex)

        if is_multi_index:
            loss_mat = activation_mat.apply(loss_multi, args=(y_app.fillna(0.0),))
        else:
            loss_mat = activation_mat.apply(loss, args=(y_app.fillna(0.0),))

        loss_mat = loss_mat.values.astype('float')

        loss_max = np.max(loss_mat)
        if is_multi_index:
            h_agg = float(len(set(y_app.index.get_level_values(0))))
        else:
            h_agg = float(len(y_app))
        nb_rules = len(self)
        eta = 1. / loss_max * np.sqrt(8. * np.log(nb_rules) / float(h_agg))

        pi_mat = pd.DataFrame(index=range(int(h_agg)),
                              columns=range(nb_rules))

        for i in range(int(h_agg)):
            if i == 0:
                pi_mat.iloc[i] = 1. / nb_rules
            else:
                w_past = pi_mat.iloc[i - 1]
                second_loss = loss_mat[i] - sum(loss_mat[1] * w_past)
                pi_mat.iloc[i] = w_past * np.exp(-eta * loss_mat[i] -
                                                 pow(eta * second_loss, 2))
                pi_mat.iloc[i] /= sum(pi_mat.iloc[i])

        weight_vect = pi_mat.mean(axis=0).values
        weight_vect /= sum(weight_vect)

        return weight_vect

    def calc_weight_agg(self, y_app, agg_method, learning):
        """
        Computes the weights for a given aggregation method
        """
        weight_vect = []
        if agg_method == 'Wexp':
            weight_vect = self.calc_weight_wexp(y_app, learning)

        elif agg_method == 'BOA':
            weight_vect = self.calc_weight_boa(y_app, learning)

        else:
            learning.message('Choose aggregation belong [Equi, Wexp, BOA]', 3)

        return weight_vect

    def calc_pred(self, ymean, ystd, learning, in_sample=False):
        """
        Computes the prediction vector

        Warnings: It's done with the equally weighted aggregation
        """
        weight_vect = self.calc_weight(learning=learning,
                                       in_sample=in_sample)
        pred_vect = sum(map(lambda rg: weight_vect *
                            (rg.calc_activation(learning=learning,
                                                in_sample=in_sample) *
                             ystd *
                             transfert(rg.get_param('pred'))),
                            self))
        return pred_vect + ymean

    def calc_pred_mat(self, learning, in_sample=False,
                      nb_jobs=None, add_pbar=False):
        pred_mat = pd.DataFrame()

        if nb_jobs is None:
            nb_jobs = learning.get_param('nb_jobs')

        out_q = mlp.Queue()
        jobs = []
        data_len = len(self) / nb_jobs
        for i in range(nb_jobs):
            if i == nb_jobs - 1:
                sub_rules_list = self[data_len * i:]
            else:
                sub_rules_list = self[data_len * i:data_len*(i+1)]

            p = mlp.Process(target=do_pred_mat, args=(learning, sub_rules_list,
                                                      in_sample, out_q, i))
            p.start()
            jobs.append(p)
            learning.message('Process %s starts' % str(i))

        if add_pbar:
            pbar = tqdm.tqdm(range(len(self)), ncols=60, desc='Calc Pred Mat')
        else:
            pbar = None

        i = 0
        while any(map(lambda pr: pr.is_alive(), jobs)) or out_q.empty() is False:
            if out_q.empty() is False:
                output = out_q.get()

                if type(output) == int:
                    # jobs[output].terminate()
                    jobs[output].join()
                    learning.message('Process %s is terminated' % str(output))

                elif output.__class__ == tuple:
                    if hasattr(output[0], '_name'):
                        name = output[0].get_param('name')
                    else:
                        name = 'R'+str(i)
                    pred_mat[name] = output[1]
                    i += 1
                    if pbar is not None:
                        pbar.update()
                else:
                    learning.message('Error with output %s' % str(output[0]), 2)

        if pbar is not None:
            pbar.close()

        return pred_mat

    def calc_activation(self, learning=None):
        active_vect = sum(map(lambda rg:
                              rg.calc_activation(learning=learning),
                              self))
        active_vect = 1 * active_vect.astype('bool')
        return active_vect

    def calc_coverage(self, learning=None):
        if len(self) > 0:
            active_vect = self.calc_activation(learning)
            cov = sum(active_vect) * 1.0 / len(active_vect)
        else:
            cov = 0.0
        return cov

    def calc_crit(self, y, ymean, ystd,
                  var_horizon, method='mse_function',
                  n=252, pen='std', learning=None):

        pred_vect = self.calc_pred(ymean, ystd, learning)

        if method == 'mse_function':
            crit = mse_function(pred_vect, y)
            crit_min = np.nan
            crit_max = np.nan
            nb_activation = np.nan
            out = False
            tuple_rez = (crit, crit_min, crit_max, nb_activation, out)

        elif method == 'mae_function':
            crit = mae_function(pred_vect, y)
            crit_min = np.nan
            crit_max = np.nan
            nb_activation = np.nan
            out = False
            tuple_rez = (crit, crit_min, crit_max, nb_activation, out)

        elif method == 'pengain':
            if data_utils.get_nb_assets(y) == 1:
                tuple_rez = pen_gain(pred_vect, y, n, var_horizon, pen)

            else:
                tuple_rez = pen_gain_multi(pred_vect, y, n, pen)
        else:
            raise 'Method %s unknown' % method

        return tuple_rez

    def make_clustering(self, olearning, nb_clusters):
        pred_mat = self.calc_pred_mat(learning=olearning, in_sample=True).T

        clusters = hac.fclusterdata(X=pred_mat, t=nb_clusters,
                                    metric=dist,
                                    criterion='maxclust',
                                    method='ward')

        olearning.message('Loop for attribution of cluster', 1)
        for i, rg in enumerate(self):
            rg.set_params(cluster=clusters[i])

    def extract_from_cluster(self, olearning, nb_clusters):
        self.make_clustering(olearning, nb_clusters)
        new_rs = RuleSet([])

        olearning.message('Extraction from cluster', 1)
        for cl in range(nb_clusters):
            sub_rs = self.extract('cluster', cl)
            if len(sub_rs) > 0:
                if olearning.get_param('maximized'):
                    id_rules = np.argmax(sub_rs.get_rules_param('crit'))
                else:
                    id_rules = np.argmin(sub_rs.get_rules_param('crit'))
                new_rs.append(sub_rs[id_rules])

        return new_rs

    def predict(self, y_app, x, ymean, ystd, agg_method, learning=None, index=None):
        """
        Computes the prediction vector for a given X and a given aggregation method
        """

        if agg_method == 'Equi':
            weight_vect = self.calc_weight_equi(x, learning, index)
            pred_vect = sum(map(lambda rg: weight_vect *
                                rg.predict(x, learning, index), self))

        else:
            weight_vect = self.calc_weight_agg(y_app, agg_method, learning)

            renorm_vect = sum(map(lambda rg, pi: pi * rg.calc_activation(x, learning, index),
                                  self, weight_vect))

            pred_vect = sum(map(lambda rg, pi: pi * rg.predict(x, learning, index),
                                self, weight_vect))
            pred_vect /= renorm_vect
            pred_vect = np.nan_to_num(pred_vect)

        return pred_vect * ystd + ymean

    def find_in_history(self, x, x_old, y_old):
        """
        find x in X_old and get y_old
        """
        pass

    def calc_active_pred(self, y_app, learning):
        active_vect = self.calc_activation(learning=learning)
        pred = sum(active_vect * y_app)
        pred /= sum(active_vect)
        return pred

    def del_activation(self):
        for rg in self:
            if hasattr(rg, '_activation'):
                rg.set_params(activation=None)

    """------   Getters   -----"""
    def get_active_rules(self, x):
        rules = filter(lambda rg: rg.calc_activation(X=x) == 1, self)
        return RuleSet(rules)

    def get_rules_param(self, param):
        return map(lambda rg: rg.get_param(param), self)

    def get_rules(self):
        return self._rules

    def get_rules_name(self):
        try:
            return map(lambda rg: rg.get_param('name'), self)
        except AssertionError:
            map(lambda rg, rg_id: rg.make_name(rg_id), self, range(len(self)))
            return map(lambda rg: rg.get_param('name'), self)

    """------   Setters   -----"""
    def set_rules(self, rules):
        assert type(rules) == list, 'Must be a list object'
        self._rules = rules


class Learning(BaseEstimator):
    """
    classdocs
    """

    def __init__(self, logger=None, **parameters):
        """
        AdLeanr is an learning object.

        Parameters
        ----------
        path_in : {string type}
                  Absolute path to the data. It can be a csv file or
                  a directory

        path_index : {string type} default None
                     Absolute path to the csv Index file if it's
                     compulsory

        normalized : {boolean type} default True
                    To choose if y is already standardized variable
                    i.e E[y] = 0 and Var(y) = 1

        maximized : {boolean type} default False
                    To choose if the criterion must be maximized
                    Otherwise it will be minimized

        method : {string type} default mse_function
                 Choose among the mse_function and MAE criterion

        nb_buckets : {int type} default 20
                    Choose the number a bucket for the discretization

        cp : {int type} default 2
             Choose the maximal complexity of one rule

        covmin : {float type such as 0 < covmin < 1} default 0.05
                 Choose the minimal coverage of one rule

        intermax : {float type such as 0 < intermax < 1} default 0.1
                   Choose the maximal intersection rate begin a rule and
                   a current selected ruleset

        ists : {bool type} default True
               Choose if the data are time series or not

        nb_jobs : {int type} default 6
                  Select the number of CPU used

        lin_select : {boolean type} default False
                     Do a linear selection before the minimum/maximum
                     contrast selection to decrease the length of the ruleset.
                     The linear model is an Elastic Net regression
                     (L1 and L2 penalty) with positives coefficients

        alpha : {float type such as 0 < alpha < 1} default 0.05
                Parameter of the Elastic Net regression

        l1_ratio : {float type such as 0 < l1_ratio < 1} default 0.5
                   Parameter of the Elastic Net regression

        sep_select : {boolean type} default True
                     Choose if the positive and negative rules
                     are selected separately
        """
        self._selected_rs = RuleSet([])
        self._rs = RuleSet([])
        self._bins = dict()
        self._ymean = 0
        self._ystd = 1
        self._cpname = socket.gethostname()

        self.logger = logger

        self._n = 252
        self._ists = True
        self._nb_buckets = 10
        self._covmin = 0.05
        self._covmax = 0.50
        self._normalized = False
        self._reduced = False
        self._centred = False
        self._maximized = True
        self._nb_clusters = 50
        self._block_size = 0

        self._calcmethod = 'pengain'
        self._pen = 'std'
        self._sinicrit = 'zscore'
        self._selectmethod = 'fast'
        self._aggmethod = 'Equi'
        self._fullselection = False
        self._sep_select = False

        self._sep_select = True
        self._path_index = ''

        self._lin_select = False
        self._nb_jobs = mlp.cpu_count() - 2
        self._intermax = 0.8

        self.name = None

        for arg, val in parameters.items():
            setattr(self, '_' + arg, val)

        # self.find_data_path()

    def __str__(self):
        return self.name

    def make_name(self):
        end = self.get_param('dtend')
        start = self.get_param('dtstart')
        var_name = self.get_param('target')
        if start != '':
            name = var_name + '_' + start.strftime("%d-%m-%Y") + '-' + end.strftime("%d-%m-%Y")
        else:
            name = var_name + '_' + end.strftime("%d-%m-%Y")

        return name

    # def find_data_path(self):
    #     path_in = self.get_param('path_in')
    #     pathy = ''
    #     pathx = ''
    #     if os.path.isfile(path_in):
    #         pathy = path_in
    #         pathx = path_in
    #     elif os.path.isdir(path_in):
    #         if path_in[-1] != '/':
    #             path_in += '/'
    #         pathy = path_in + 'Y/'
    #         pathx = path_in + 'X/'
    #     else:
    #         self.message('Error with the path %s' % path_in, 4)
    #
    #     self.set_params(pathy=pathy)
    #     self.set_params(pathx=pathx)

    def fit(self, target, features_names=None, path_out=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        target : {string type},
                 Name of the y variable

        features_names : {array-like}, shape = [n_features], optional
            Array of features names.

        path_out :

        Returns
        -------
        return : {Learning type}
                 return nothing but the self is different
        """
        if features_names is None:
            features_names = self.load_xlist()
            """DEBUG"""
            # features_names = ['FX_CTdelta5d']

        self.message('There are %s variables' % str(len(features_names)), 1)
        self.set_params(target=target)

        self.set_params(features_names=features_names)

        if self.name is None:
            self.name = self.make_name()

        if hasattr(self, 'logger') is False or self.logger is None:
            self.logger = init_logger(path_out, target)

        y = self.load_target(target)

        if self.get_param('normalized') is True:
            y -= y.mean()
            y /= y.std()
            self.set_params(ymean=0)
            self.set_params(ystd=1)
        elif self.get_param('centered') is True:
            y -= y.mean()
            self.set_params(ystd=y.std())
            self.set_params(ymean=0)
        elif self.get_param('reduced') is True:
            y /= y.nanstd()
            self.set_params(ymean=y.mean())
            self.set_params(ystd=1)
        else:
            self.set_params(ymean=y.mean())
            self.set_params(ystd=y.std())

        self.message('Ymean= %s ' % str(y.mean()), 1)
        self.message('Ystd= %s ' % str(y.std()), 1)
        self.message('And normalized is %s' % self.get_param('normalized'), 1)
        self.message('And reduced is %s' % self.get_param('reduced'), 1)
        self.message('And centered is %s' % self.get_param('centered'), 1)

        self.set_params(y=y)

        self.find_rules(path_out)  # works in columns not in lines

        # self.write_results()

    def find_rules(self, path_out):
        """
        Find all rules for all complexity <= cp
        and for each complexity selects the best subset
        """
        complexity = self.get_param('cp')
        maximized = self.get_param('maximized')
        th_list = [0.1, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 1.0]

        assert complexity > 0, \
            'Complexity must be strictly superior to 0'

        i = 0
        while len(self.get_param('rs')) == 0:
            self.message('Seeking of CP1 rules', 1)
            self.calc_cp1(path_out)

            if len(self.get_param('rs')) == 0:
                self.set_params(th=th_list[i])
                self.message('New significant threshold %s'
                             % str(th_list[i]), 1)
                i += 1

        rules_set = self.get_param('rs')
        if len(rules_set) == 0:
            self.message('0 rules fors CP1', 3)
        else:
            self.message('%s rules for CP1' % len(rules_set), 1)

        rules_set.sort_by('crit', maximized)

        if len(self.get_param('selected_rs')) == 0:
            self.message('Selection CP1', 1)
            selected_rs = self.select_rules(1)

            self.set_params(selected_rs=selected_rs)

            self.message('%s selected rules:'
                         % len(selected_rs), 1)

            if path_out is not None:
                rules_set.to_df().to_csv(path_out + 'All_rules.csv')
                selected_rs.to_df().to_csv(path_out + 'Selected_rules.csv')

                # self.save_as_pickle(path_out, str(self), drop_activation=False)

        for cp in range(2, complexity + 1):

            rs = self.get_param('rs')  # Load the Full ruleset
            selected_rs = self.get_param('selected_rs')
            rules_set.sort_by('crit', maximized)  # sort by crit

            if any(map(lambda rg: rg.get_param('cp') == cp,
                       rs)) is False:
                self.message('Seeking of CP%s rules' % str(cp), 1)
                ruleset_cpup = self.up_complexity(cp, path_out)  # return a ruleset more complex

                if len(ruleset_cpup) > 0:
                    rules_set += ruleset_cpup  # add the new ruleset to the full one
                    self.set_params(rs=rs)  # set in into self parameters

                    self.message('%s rules for CP%s'
                                 % (len(ruleset_cpup), cp), 1)
                else:
                    break

            if any(map(lambda rg: rg.get_param('cp') == cp,
                       self.get_param('selected_rs'))) is False:
                self.message('Selection CP' + str(cp), 1)
                # Select the complex rules
                ruleset_cpup_selected = self.select_rules(cp)

                selected_rs += ruleset_cpup_selected

                self.set_params(selected_rs=selected_rs)

            if path_out is not None:
                rules_set.to_df().to_csv(path_out + 'All_rules.csv')
                selected_rs.to_df().to_csv(path_out + 'Selected_rules.csv')

                # self.save_as_pickle(path_out, str(self), drop_activation=False)

    def calc_rules(self, x, values, feature_name, bmin, close_var, out_q=None):
        """
        Compute all rules for a given feature variable and
        a given borne min.
        """
        var_horizon = self.get_param('varhorizon')
        method = self.get_param('calcmethod')
        n = self.get_param('n')
        y = self.get_param('y')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        cov_min = self.get_param('covmin')
        pen = self.get_param('pen')
        path_acceptable = self.get_param('path_acceptable')
        path_desciption = self.get_param('path_desciption')
        th = self.get_param('th')

        j = values.tolist().index(bmin)
        for bmax in values[j:]:
            try:
                conditions = RuleConditions(features_names=[feature_name],
                                            bmin=[bmin],
                                            bmax=[bmax],
                                            xmax=[max(values)],
                                            xmin=[min(values)],
                                            values=values)
                rule = Rule(conditions)

                rule.calc_stats(x=x, y=y, ymean=ymean, ystd=ystd,
                                var_horizon=var_horizon, method=method,
                                n=n, cov_min=cov_min, pen=pen,
                                path_acceptable=path_acceptable,
                                path_desciption=path_desciption,
                                close_var=close_var,
                                th=th)

                if rule.get_param('out') is False:
                    out_q.put(rule)
                else:
                    self.message('Bad rule %s' % str(rule))
                    reason = rule.get_param('reason')

                    if reason == 'Criterium':
                        c_min = rule.get_param('crit_min')
                        c_max = rule.get_param('crit_max')
                        self.message('Bad criterion min:%s, max:%s'
                                     % (str(c_min), str(c_max)))
                    else:
                        self.message(reason)

            except Exception as e:
                ligne = traceback.extract_tb(sys.exc_traceback)[0][1]
                fonction = traceback.extract_tb(sys.exc_traceback)[0][2]
                erreur_str = str(e) + '\n'
                erreur_str += 'Error in the function ' + str(fonction) + ', line ' + str(ligne) + '.'
                self.message('Error with rule %s' % str(feature_name), 2)
                self.message('Message is: %s' % erreur_str, 2)

    def calc_var(self, feature_name, close_var, out_q=None):
        """
        Compute all rules on a given feature variable
        """
        try:
            xcol = self.load_var(feature_name)

            if xcol is not None:
                notnan_vect = np.extract(np.isfinite(xcol), xcol)
                values = np.array(np.sort(list(set(notnan_vect))), dtype=np.float64)

                if len(values) == 0:
                    self.message('Bad Variables %s 0 different value' % feature_name)
                else:
                    self.message('%s has %s different values' % (feature_name,
                                                                 str(len(values))))

                    map(lambda bmin: self.calc_rules(xcol, values,
                                                     feature_name,
                                                     bmin, close_var,
                                                     out_q), values)

        except Exception as e:
            ligne = traceback.extract_tb(sys.exc_traceback)[0][1]
            fonction = traceback.extract_tb(sys.exc_traceback)[0][2]
            erreur_str = str(e) + '\n'
            erreur_str += 'Error in the function ' + str(fonction) + ', line ' + str(ligne) + '.'
            self.message('Error with the variable %s' % str(feature_name), 2)
            self.message('Message is: %s' % erreur_str, 2)

        out_q.put(feature_name)

    def calc_cp1(self, path_out):
        """
        Compute all rules of complexity one and keep the best
        """
        features_list = self.get_param('features_names')
        nb_jobs = self.get_param('nb_jobs')
        out_q = mlp.Queue()
        ruleset = self.get_param('rs')
        time_serie = self.get_param('ists')

        close_name = self.get_param('closename')

        if close_name is not None:
            close_var = data_utils.load_var(path_out, close_name, time_serie)
        else:
            close_var = None

        jobs = []
        data_len = len(features_list) / nb_jobs

        """ DEBUG """
        # map(lambda var: self.calc_var(var, close_var, out_q),
        #     tqdm.tqdm(features_list))
        #
        # while out_q.empty() is False:
        #     output = out_q.get()
        #
        #     if output.__class__ == Rule:
        #         ruleset.append(output)
        #
        #     else:
        #         self.logger.warn('Error with output %s' % output)

        """ Multiprocessing """
        for i in range(nb_jobs):
            if i == nb_jobs - 1:
                vars_list = features_list[data_len * i:]
            else:
                vars_list = features_list[data_len * i:data_len * (i + 1)]
            p = mlp.Process(target=calc_vars, args=(self, vars_list,
                                                    close_var, i, out_q))
            jobs.append(p)
            p.start()
            self.message('Process %s starts' % str(i))

        pbar = tqdm.tqdm(range(len(features_list)), ncols=60, desc='Calc CP1')

        bins_dict = {}

        while any(map(lambda pr: pr.is_alive(), jobs)) or out_q.empty() is False:
            if out_q.empty() is False:
                output = out_q.get()

                if type(output) == int:
                    # jobs[output].terminate()
                    jobs[output].join()
                    self.message('Process %s is terminated' % str(output))

                elif output.__class__ == Rule:
                    ruleset.append(output)
                elif type(output) == dict:
                    bins_dict.update(output)
                elif type(output) == str:
                    pbar.update()
                else:
                    self.message('Error with output %s' % output, 2)

        self.set_params(bins=bins_dict)
        pbar.close()
        ruleset.sort_by('crit', self.get_param('maximized'))

        self.set_params(rs=ruleset)

    def up_complexity(self, cp, path_out):
        """
        Returns a ruleset of rules with complexity=cp
        """
        y = self.get_param('y')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        method = self.get_param('calcmethod')
        n = self.get_param('n')
        cov_min = self.get_param('covmin')
        var_horizon = self.get_param('varhorizon')
        nb_jobs = self.get_param('nb_jobs')
        pen = self.get_param('pen')
        ists = self.get_param('ists')

        close_name = self.get_param('closename')

        if close_name is not None:
            close_var = data_utils.load_var(path_out, close_name, ists)
        else:
            close_var = None

        rules_list = self.find_candidates(cp, cov_min)

        ruleset = RuleSet([])

        if len(rules_list) > 0:
            self.message('Evaluations of rules', 1)

            out_q = mlp.Queue()

            jobs = []
            data_len = len(rules_list) / nb_jobs

            for i in range(nb_jobs):
                if i == nb_jobs - 1:
                    sub_rules_list = rules_list[data_len * i:]
                else:
                    sub_rules_list = rules_list[data_len * i:data_len * (i + 1)]

                p = mlp.Process(target=calc_upcp, args=(self, sub_rules_list,
                                                        y, ymean, ystd, var_horizon,
                                                        method, n, cov_min, pen,
                                                        close_var, i, out_q))
                jobs.append(p)
                p.start()
                self.message('Process %s starts' % str(i))

            pbar = tqdm.tqdm(range(len(rules_list)), ncols=60,
                             desc='Calc CP%s' % cp)

            while any(map(lambda pr: pr.is_alive(), jobs)) or out_q.empty() is False:
                if out_q.empty() is False:
                    output = out_q.get()

                    if type(output) == int:
                        # jobs[output].terminate()
                        jobs[output].join()
                        self.message('Process %s is terminated' % str(output))

                    elif output.__class__ == Rule:
                        ruleset.append(output)
                        pbar.update()
                    elif output is None:
                        pbar.update()
                    else:
                        self.message('Error with output %s' % output, 2)

            pbar.close()
            rules_list = filter(lambda rg: rg.get_param('out') is False,
                                ruleset.get_rules())

            return RuleSet(rules_list)

        else:
            self.message('No candidats for CP%s' % str(cp), 1)
            return ruleset

    def find_candidates(self, cp, cov_min):
        """
        Returns the intersection of all suitable rules
        for a given complexity (cp) and a min coverage (cov_min)
        """
        rs = self.get_param('rs')
        nb_clusters = self.get_param('nb_clusters')

        rs_cp1 = rs.extract_cp(1)
        cov_th = pow(cov_min, 1. / cp)

        if self.get_candidate() is True or cp > 2:
            if nb_clusters > 0:
                rs_cp1 = self.select_from_cluster(rs_cp1, cov_th)
            else:
                rs_cp1 = select_from_bend(rs_cp1, cov_th,
                                          300.0 / cp)
            if len(rs_cp1) > 0:
                rs_cp1 = self.add_complementary(rs_cp1)
            else:
                return []

        rs_candidate = rs.extract_cp(cp-1)
        cov_th = pow(cov_min, float(cp - 1) / cp)

        if self.get_candidate() is True or cp > 2:
            if nb_clusters > 0:
                rs_candidate = self.select_from_cluster(rs_candidate, cov_th)
            else:
                rs_candidate = select_from_bend(rs_candidate, cov_th,
                                                300.0 / cp)

        nb_candidate = len(rs_candidate)
        self.message('Finding rules more complex', 1)
        self.message('We have %s CP1 rules'
                     % str(len(rs_cp1)), 1)
        self.message('We have %s CP%s rules candidates'
                     % (str(nb_candidate), str(cp - 1)), 1)

        self.message('Tests of intersection before evaluation of CP%s'
                     % str(cp), 1)

        if len(rs_candidate) > 0:
            rules_list = self.find_complexe_rules(cp, rs_cp1,
                                                  rs_candidate)
            return rules_list
        else:
            return []

    def add_complementary(self, ruleset):
        """
        Returns the complementary ruleset of a given ruleset.
        Each rule is replaced by it(s) complementary rule(s)
        """
        cplt_rs = reduce(lambda a, b: a + b,
                         map(lambda rg: rg.complement(),
                             ruleset))
        cplt_rs = filter(None, cplt_rs)

        # To drop duplicate
        cplt_rs = list(set(cplt_rs))
        cplt_rs = filter(lambda rg: rg not in ruleset, cplt_rs)

        out_q = mlp.Queue()
        nb_jobs = self.get_param('nb_jobs')
        data_len = len(cplt_rs) / nb_jobs

        jobs = []
        for i in range(nb_jobs):
            if i == nb_jobs - 1:
                sub_ruleset = cplt_rs[data_len * i:]
            else:
                sub_ruleset = cplt_rs[data_len * i:data_len * (i + 1)]

            p = mlp.Process(target=calc_clpt, args=(self, sub_ruleset,
                                                    i, out_q))
            jobs.append(p)
            p.start()
            self.message('Process %s starts' % str(i))

        pbar = tqdm.tqdm(range(len(cplt_rs)), ncols=60,
                         desc='Calc complementary')

        while any(map(lambda pr: pr.is_alive(), jobs)) or out_q.empty() is False:
            if out_q.empty() is False:
                output = out_q.get()

                if type(output) == int:
                    # jobs[output].terminate()
                    jobs[output].join()
                    self.message('Process %s is terminated' % str(output))

                elif output.__class__ == Rule:
                    ruleset.append(output)
                    pbar.update()
                elif output is None:
                    pbar.update()
                else:
                    self.message('Error with output %s' % output, 2)

        pbar.close()
        return ruleset

    def find_complexe_rules(self, cp, ruleset_cp1, ruleset_candidate):
        nb_jobs = self.get_param('nb_jobs')
        out_q = mlp.Queue()

        jobs = []
        rules_list = []
        data_len = len(ruleset_candidate) / nb_jobs

        for i in range(nb_jobs):
            if i < nb_jobs - 1:
                candidates_list = ruleset_candidate[data_len * i:data_len * (i + 1)]
            else:
                candidates_list = ruleset_candidate[data_len * i:]

            if cp == 2:
                cp1_list = ruleset_cp1[data_len * i:]
            else:
                cp1_list = copy.copy(ruleset_cp1)

            p = mlp.Process(target=find_upcp, args=(self, candidates_list,
                                                    cp1_list, cp, i, out_q))

            jobs.append(p)
            p.start()
            self.message('Process %s starts' % str(i))

        pbar = tqdm.tqdm(range(len(ruleset_candidate)),
                         ncols=60, desc='Tests for CP%s' % str(cp))

        while any(map(lambda pr: pr.is_alive(), jobs)) or out_q.empty() is False:
            if out_q.empty() is False:
                output = out_q.get()

                if type(output) == int:
                    # jobs[output].terminate()
                    jobs[output].join()
                    self.message('Process %s is terminated' % str(output))

                elif type(output) == list:
                    rules_list += output
                    pbar.update()

                elif output is None:
                    pbar.update()

                else:
                    self.message('Error with output %s' % output, 2)

        pbar.close()

        rules_list = filter(None, rules_list)  # to drop bad rules
        rules_list = list(set(rules_list))  # to drop duplicates
        return rules_list

    def select_rules(self, cp):
        """
        Returns a subset of a given ruleset. This subset is seeking by
        elastic net regression.
        """
        rs = self.get_param('rs')
        nb_clusters = self.get_param('nb_clusters')
        cov_max = self.get_param('covmax')
        sub_rs = rs.extract_cp(cp)

        if len(sub_rs) > 0:
            if nb_clusters > 0:
                rules_set = self.select_from_cluster(sub_rs, cov_max=cov_max)
            else:
                rules_set = sub_rs.extract_least('cov', cov_max)

            separate_selection = self.get_param('sep_select')
            select_method = self.get_param('selectmethod')

            if select_method == 'fast':
                select_func = fast_selection
            else:
                select_func = optimize_selection

            if separate_selection:
                rs_pos = rules_set.extract_greater('pred', 0)
                rs_neg = rules_set.extract_least('pred', 0)

                out_q = mlp.Queue()
                jobs = []
                if len(rs_pos) > 0:
                    self.message('%s positive rules for selection'
                                 % str(len(rs_pos)), 1)
                    p_pos = mlp.Process(target=select_func,
                                        args=(self, rs_pos,
                                              'Pos', out_q))
                    p_pos.start()
                    jobs.append(p_pos)
                else:
                    p_pos = None
                    self.message('No positive rules for selection', 1)

                if len(rs_neg) > 0:
                    self.message('%s negative rules for selection'
                                 % str(len(rs_neg)), 1)
                    p_neg = mlp.Process(target=select_func,
                                        args=(self, rs_neg,
                                              'Neg', out_q))
                    p_neg.start()
                    jobs.append(p_neg)
                else:
                    p_neg = None
                    self.message('No negative rules for selection', 1)

                selected_rs = RuleSet([])
                while any(map(lambda pr: pr.is_alive(), jobs)):
                    if out_q.empty() is False:
                        output = out_q.get()
                        if output.__class__ == RuleSet:
                            selected_rs.extend(output)
                        elif output == 'Pos' and p_pos is not None:
                            p_pos.join()
                            self.message('Selection of positive is over', 1)
                        elif output == 'Neg' and p_neg is not None:
                            p_neg.join()
                            self.message('Selection of negative is over', 1)

            else:
                name = 'CP' + str(cp)
                selected_rs = select_func(self, rules_set, name)

            self.message('Selection is over', 1)
            self.message('%s CP%s rules selected'
                         % (len(selected_rs), cp), 1)
        else:
            selected_rs = RuleSet([])

        return selected_rs

    def select_from_cluster(self, rs, cov_min=0.0, cov_max=1.0):

        out_q = mlp.Queue()
        rs_pos = rs.extract_greater('pred', 0.0)
        rs_neg = rs.extract_least('pred', 0.0)

        jobs = []
        if len(rs_pos) > 0:
            self.message('%s positive rules to clusterize'
                         % str(len(rs_pos)), 1)
            p_pos = mlp.Process(target=do_clustering, args=(self, rs_pos,
                                                            cov_min, cov_max,
                                                            out_q, 'pos'))
            p_pos.start()
            jobs.append(p_pos)
        else:
            p_pos = None
            self.message('No positive rules for clustering', 1)

        if len(rs_neg) > 0:
            self.message('%s negative rules to clusterize'
                         % str(len(rs_neg)), 1)
            p_neg = mlp.Process(target=do_clustering, args=(self, rs_neg,
                                                            cov_min, cov_max,
                                                            out_q, 'neg'))
            p_neg.start()
            jobs.append(p_neg)
        else:
            p_neg = None
            self.message('No negative rules for clustering', 1)

        self.message('Wait during clustering...', 1)
        animation = '|/-\\'
        idx = 0

        selected_rs = RuleSet([])
        while any(map(lambda pr: pr.is_alive(), jobs)):
            sys.stdout.write(animation[idx % len(animation)] + "\r")
            idx += 1
            time.sleep(0.1)
            sys.stdout.flush()
            if out_q.empty() is False:
                output = out_q.get()
                if output.__class__ == RuleSet:
                    selected_rs.extend(output)
                elif output == 'pos' and p_pos is not None:
                    p_pos.join()
                    self.message('Clustering of positive is over', 1)
                elif output == 'neg' and p_neg is not None:
                    p_neg.join()
                    self.message('Clustering of negative is over', 1)

        self.message('Clustering is over', 1)

        return selected_rs

    def predict(self, x=None, agg_method='Equi', index=None):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        application of the selected ruleset on X.

        Parameters
        ----------
        x : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        agg_method : string-type

        index : pandas index type

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        if agg_method != 'Equi':
            y_app = self.get_param('y')
        else:
            y_app = None

        if x is not None:
            x_copy = copy.copy(x)
            x_copy.apply(lambda col: self.discretize(col), axis=1)
        else:
            x_copy = None

        ruleset = self.get_param('selected_rs')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')

        pred_vect = ruleset.predict(y_app, x_copy, ymean, ystd,
                                    agg_method, self, index)

        return pred_vect

    def score(self, x, y, sample_weight=None, agg_method='Equi'):
        """
        Returns the coefficient of determination R^2 of the prediction
        if y is continuous. Else if y in {0,1} then Returns the mean
        accuracy on the given test data and labels {0,1}.

        Parameters
        ----------
        x : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        agg_method : string type

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y in R.

            or

        score : float
            Mean accuracy of self.predict(X) wrt. y in {0,1}
        """

        x_copy = copy.copy(x)

        pred_vect = self.predict(x_copy, agg_method)

        nan_val = np.argwhere(np.isnan(y))
        if len(nan_val) > 0:
            pred_vect = np.delete(pred_vect, nan_val)
            y = np.delete(y, nan_val)

        if len(set(y)) == 2:
            th_val = (min(y) + max(y)) / 2.0
            pred_vect = np.array(map(lambda p: min(y) if p < th_val else max(y),
                                     pred_vect))
            return accuracy_score(y, pred_vect)
        else:
            return r2_score(y, pred_vect, sample_weight=sample_weight,
                            multioutput='variance_weighted')

    def write_results(self, path):
        # Writing of results
        full_rs = self.get_param('rs')
        selected_rs = self.get_param('selected_rs')
        full_rs.to_df().to_csv(r'All_rules.csv')
        selected_rs.to_df().to_csv(path + 'Selected_rules.csv')
        selected_rs.save_rules_df(path + 'rules.csv')

    def save_bins_csv(self, path_out):
        bins = self.get_param('bins')
        path_out += 'bins.csv'
        bins_df = pd.DataFrame(dict([(k, pd.Series(v))
                                     for k, v in bins.iteritems()])).T
        bins_df.to_csv(path_out)

    def save_as_pickle(self, path_out, name, drop_activation=True):
        fp = open(path_out + name, 'wb')

        if hasattr(self, 'logger'):
            logger_temp = self.logger
            del self.logger

        if drop_activation:
            self.del_activation()

        cPickle.dump(self, fp)
        fp.close()

        if os.name == 'nt':
            self.message('No compression on Windows', 1)
        else:
            tar_name = name + '.tar.gz'
            dir_name = path_out.replace(' ', '\\ ')
            rep = os.system('cd %s && tar -czf %s %s' % (dir_name, tar_name, name))

            if rep == 0:
                os.system('rm %s' % (dir_name + name))
            else:
                self.message('Error to compress the pickle file', 4)

        if 'logger_temp' in locals():
            self.logger = logger_temp

    def del_activation(self):
        rs = self.get_param('rs')
        if len(rs) > 0:
            rs.del_activation()
            self.set_params(rs=rs)

        selected_rs = self.get_param('selected_rs')
        if len(selected_rs) > 0:
            selected_rs.del_activation()
            self.set_params(selected_rs=selected_rs)

    """------   Data functions   -----"""
    def load_xlist(self):
        path = self.get_param('pathx')
        is_ts = self.get_param('ists')
        xlist = []

        if os.path.isfile(path):
            xlist = data_utils.get_variables_from_file(path, is_ts)

        elif os.path.isdir(path):
            xlist = data_utils.get_variables_from_dir(path)

        else:
            self.message('Path error %s' % path, 4)

        if hasattr(self, '_regex'):
            regex = self.get_param('regex')
            for reg in regex:
                xlist = filter(lambda x: reg.upper() not in x.upper(), xlist)

        return xlist

    def load_var(self, var_names, idx_series=None):
        path = self.get_param('pathx')
        time_serie = self.get_param('ists')

        if type(var_names) == str:
            try:
                cols = data_utils.load_var(path, var_names, time_serie)
            except Exception as e:
                self.message('Error to load variable %s' % var_names, 2)
                self.message('Message is %s' % e.message, 1)
                return None
            if len(cols) > 0:
                cols = self.apply_index(cols, idx_series)
                cols = self.discretize(cols)
                cols = [cols]
            else:
                cols = None

        elif type(var_names) == list:
            cols = []
            for var in var_names:
                try:
                    col = data_utils.load_var(path, var, time_serie)
                except Exception as e:
                    self.message('Error to load variable %s' % var, 2)
                    self.message('Message %s' % e.message, 2)
                    return None

                col = self.apply_index(col, idx_series)
                col = self.discretize(col)
                cols.append(col)

        else:
            cols = None

        return cols

    def discretize(self, col):
        """
        Used to have discrete values for each series
        to avoid float

        Parameters
        ----------
        col : {array, matrix type}, shape=[n_samples, n_features]
            Features matrix

        Return
        -------
        col : {array, matrix type}, shape=[n_samples, n_features]
            Features matrix with each features values discretized
            in nb_buckets values
        """
        nb_buckets = self.get_param('nb_buckets')
        bins_dict = self.get_param('bins')
        var_name = col.name

        if len(set(col.dropna().values)) > nb_buckets and 'TIME' not in col.name:
            if var_name not in bins_dict:
                bins = data_utils.find_bins(col, nb_buckets)
                self._bins[var_name] = bins
            else:
                bins = bins_dict[var_name]

            col = data_utils.discretize(col, nb_buckets, bins)

        return col

    def load_target(self, var_name):
        path = self.get_param('pathy')

        time_serie = self.get_param('ists')
        y = data_utils.load_var(path, var_name, time_serie)

        first_date = data_utils.get_first_date(y)
        dtstart = self.get_param('dtstart')

        if type(dtstart) != int:
            if (pd.to_datetime(dtstart, dayfirst=True) < first_date or
                    pd.to_datetime(dtstart, dayfirst=True) is pd.NaT):
                self.set_params(dtstart=first_date)

        y = self.apply_index(y)

        return y

    def apply_index(self, xcol, idx_series=None):
        if idx_series is None:
            idx_series = self.get_index()

        try:
            idx_hour = self.get_param('index_hour')
            idx_min = self.get_param('index_min')
        except AttributeError:
            idx_hour = None
            idx_min = None

        if idx_series is not None:
            xcol = data_utils.apply_index(xcol, idx_series.index,
                                          idx_hour, idx_min)
        else:
            dtend = self.get_param('dtend')
            dtstart = self.get_param('dtstart')
            xcol = data_utils.take_interval(xcol, dtend, dtstart)
            self.make_index(xcol)
        return xcol

    def make_index(self, col):
        index = pd.DataFrame(index=col.index)
        self.set_params(index=index)

    def plot_rules(self, var1, var2, cp=None,
                   col_pos='red', col_neg='blue'):
        """
        Plot the rectangle activation zone of rules in a 2D plot
        the color is coresponding to the intensity of the prediction

        Parameters
        ----------
        var1 : {string type}
               Name of the first variable

        var2 : {string type}
               Name of the second variable

        cp : {int type}, optional
             Option to plot only the cp1 or cp2 rules

        col_pos : {string type}, optional,
                  Name of the color of the zone of positive rules

        col_neg : {string type}, optional
                  Name of the color of the zone of negative rules

        -------
        Draw the graphic
        """

        selected_rs = self.get_param('selected_rs')
        nb_buckets = self.get_param('nb_buckets')

        if type(cp) == int:
            bool_vect = map(lambda rule: rule.get_param('cp') == cp,
                            selected_rs)
            extract = compress(selected_rs, bool_vect)
            sub_rs = RuleSet(list(extract))
        else:
            sub_rs = selected_rs

        plt.plot()

        for rg in sub_rs:
            rg_condition = rg.conditions

            var = rg_condition.get_param('features_names')
            bmin = rg_condition.get_param('bmin')
            bmax = rg_condition.get_param('bmax')
            cp_rg = rg.get_param('cp')

            if rg.get_param('pred') > 0:
                hatch = '/'
                facecolor = col_pos
                alpha = min(1, abs(rg.get_param('pred')) / 2.0)
            else:
                hatch = '\\'
                facecolor = col_neg
                alpha = min(1, abs(rg.get_param('pred')) / 2.0)

            if cp_rg == 1:
                if var == var1:
                    p = patches.Rectangle((bmin - 0.05, 0),  # origin
                                          (bmax - bmin) + 0.05,  # width
                                          nb_buckets - 1,  # height
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)

                elif var == var2:
                    p = patches.Rectangle((0, bmin - 0.05),
                                          nb_buckets - 1,
                                          (bmax - bmin) + 0.05,
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)

            elif cp_rg == 2:
                if var[0] == var1 and var[1] == var2:
                    p = patches.Rectangle((bmin[0], bmin[1]),
                                          (bmax[0] - bmin[0]),
                                          (bmax[1] - bmin[1]),
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)

                elif var[1] == var1 and var[0] == var2:
                    p = patches.Rectangle((bmin[1], bmin[0]),
                                          (bmax[1] - bmin[1]),
                                          (bmax[0] - bmin[0]),
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)

        if cp is None:
            plt.gca().set_title('Rules activations')
        else:
            plt.gca().set_title('Rules cp%s activations' % str(cp))

        plt.gca().axis([-0.5, nb_buckets - 0.5, -0.5, nb_buckets - 0.5])

    def plot_pred(self, x, y, var1, var2, cmap=False,
                  vmin=None, vmax=None, add_points=True,
                  add_score=False, agg_method='Equi'):
        """
        Plot the prediction zone of rules in a 2D plot

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape=[n_samples, n_features]
            Features matrix, where n_samples in the number of samples and
            n_features is the number of features.

        y : {array-like}, shape=[n_samples]
            Target vector relative to X

        var1 : {int type}
               Column number of the first variable

        var2 : {int type}
               Column number of the second variable

        cmap : {colormap object}, optional
               Colormap used for the graphic

        vmax, vmin : {float type}, optional
                     Parameter of the range of the colorbar

        add_points: {boolean type}, optional
                    Option to add the discret scatter of y

        add_score : {boolean type}, optional
                    Option to add the score on the graphic

        agg_method : string type

        -------
        Draw the graphic
        """

        nb_buckets = self.get_param('nb_buckets')

        x1 = self.discretize(x[var1])
        x2 = self.discretize(x[var2])

        xx, yy = np.meshgrid(range(nb_buckets),
                             range(nb_buckets))

        if cmap is None:
            cmap = plt.cm.bwr  # @UndefinedVariable

        z = self.predict(np.c_[np.round(xx.ravel()),
                               np.round(yy.ravel())],
                         agg_method=agg_method)

        if vmin is None:
            vmin = min(z)
        if vmax is None:
            vmax = max(z)

        z = z.reshape(xx.shape)

        plt.contourf(xx, yy, z, cmap=cmap, alpha=.8, vmax=vmax, vmin=vmin)

        if add_points:

            df = pd.DataFrame()
            df['Y'] = y
            df['var1'] = x1
            df['var2'] = x2

            area = map(lambda b: map(lambda a: df.loc[(df['var1'] == a) &
                                                      (df['var2'] == b)]['Y'].mean(),
                                     range(nb_buckets)), range(nb_buckets))

            area_len = map(lambda b: map(lambda a: len(df.loc[(df['var1'] == a) &
                                                              (df['var2'] == b)]['Y']) * 10,
                                         range(nb_buckets)), range(nb_buckets))

            plt.scatter(xx, yy, c=area, s=area_len, alpha=0.5,
                        cmap=cmap, vmax=vmax, vmin=vmin)

        plt.title('Learning %s' % agg_method)

        if add_score:
            score = self.score(x, y, agg_method=agg_method)
            plt.text(nb_buckets - .70, .08, ('%.2f' % score).lstrip('0'),
                     size=18, horizontalalignment='right')

        plt.axis([-0.5, nb_buckets - 0.5, -0.5, nb_buckets - 0.5])
        plt.colorbar()

    def plot_counter_variables(self):
        rs = self.get_param('selected_rs')
        counter = get_variables_count(rs)

        df = pd.DataFrame(columns=['Variable', 'Count'])

        x_labels = counter.keys()

        df['Variable'] = x_labels
        df['Count'] = counter.values()

        df.sort_values(by='Count', inplace=True, ascending=False)

        f = plt.figure()
        ax = plt.subplot()

        g = sns.barplot(y='Variable', x='Count', data=df, ax=ax, ci=None)
        g.set(xlim=(0, df['Count'].max() + 1), ylabel='Variable', xlabel='Count')

        return f

    def plot_counter(self):
        """
        Function plots a graphical counter of varaibles used in rules.
        """

        nb_buckets = self.get_param('nb_buckets')
        counter = self.make_count_matrice()

        x_labels = map(lambda i: str(i), range(nb_buckets))
        y_labels = counter.index

        counter.index = y_labels
        counter.columns = x_labels

        f = plt.figure()
        ax = plt.subplot()

        g = sns.heatmap(counter, cmap='Reds', linewidths=.05, ax=ax)
        g.xaxis.tick_top()
        plt.yticks(rotation=0)

        return f

    def plot_dist(self, cp=0, metric=dist):
        """
        Function plots a graphical correlation of rules.
        """

        rs = self.get_param('selected_rs')
        if cp > 0:
            rs = rs.extract_cp(cp)

        rules_names = rs.get_rules_name()

        pred_mat = rs.calc_pred_mat(self, in_sample=True)
        pred_mat = pred_mat.reindex(columns=rules_names)
        # pred_mat = pred_mat.replace(0.0, np.nan)

        dist_vect = scipy_dist.pdist(pred_mat.T.values, metric=metric)
        dist_mat = scipy_dist.squareform(dist_vect)

        # Set up the matplotlib figure
        f = plt.figure()
        ax = plt.subplot()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(dist_mat, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        vmax = np.max(dist_mat)
        vmin = np.min(dist_mat)
        # center = np.mean(dist_mat)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(dist_mat, mask=mask, cmap=cmap, ax=ax,
                    vmax=vmax, vmin=vmin, center=0.0,
                    square=True, xticklabels=rules_names,
                    yticklabels=rules_names)

        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        return f

    def plot_intensity(self):
        """
        Function plots a graphical counter of varaibles used in rules.
        """

        counter = self.make_count_matrice()
        intensity = self.make_count_matrice(True)

        nb_buckets = self.get_param('nb_buckets')

        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.divide(intensity.values, counter.values)

        x_labels = map(lambda i: str(i), range(nb_buckets))
        y_labels = counter.index

        intensity_mat = pd.DataFrame(val, index=y_labels, columns=x_labels)
        intensity_mat = intensity_mat.replace([np.inf, -np.inf], np.nan)
        intensity_mat = intensity_mat.fillna(0.0)

        f = plt.figure()
        ax = plt.subplot()

        g = sns.heatmap(intensity_mat, cmap='bwr',
                        linewidths=.05, ax=ax, center=0.0)
        g.xaxis.tick_top()
        plt.yticks(rotation=0)

        return f

    def plot_sign(self):
        rs = self.get_param('selected_rs')

        df = pd.DataFrame(index=range(len(rs)), columns=['Pred', 'Cp', 'Sign'])
        df['Pred'] = map(lambda rg: abs(rg.get_param('pred')), rs)
        df['Cp'] = map(lambda rg: rg.get_param('cp'), rs)
        df['Sign'] = map(lambda rg: 'Pos' if rg.get_param('pred') > 0 else 'Neg', rs)

        f = plt.figure()
        ax = plt.subplot()

        g = sns.factorplot(x='Cp', y='Pred', hue='Sign', data=df,
                           kind='bar', legend=False, ax=ax, ci=None)

        g.despine(left=True)
        ax.set(xlabel='Complexity', ylabel='Conditional Expectation')
        ax.legend(loc='upper left')

        return f

    def make_count_matrice(self, add_pred=False):
        selected_rs = self.get_param('selected_rs')
        rules_list = filter(lambda rl: 'TIME' not in rl.get_vars(), selected_rs)
        rs = RuleSet(rules_list)

        nb_buckets = self.get_param('nb_buckets')

        counter = get_variables_count(rs)
        cols = sorted(counter, key=counter.get, reverse=True)

        count_mat = pd.DataFrame(0, columns=cols, index=range(nb_buckets))

        for rg in rs:
            cd = rg.conditions
            var_name = cd.get_param('features_names')
            bmin = cd.get_param('bmin')
            bmax = cd.get_param('bmax')

            for j in range(len(var_name)):
                for b in range(int(bmin[j]), int(bmax[j]) + 1):
                    if add_pred:
                        count_mat[var_name[j]].iloc[b] += rg.get_param('pred')
                    else:
                        count_mat[var_name[j]].iloc[b] += 1

        return count_mat.T

    def message(self, message, level=0):
        if hasattr(self, 'logger') is False:
            print message
        else:
            if level == 0:
                self.logger.debug(message)

            elif level == 1:
                self.logger.info(message)

            elif level == 2:
                self.logger.warning(message)
                warnings.warn(message)

            elif level == 3:
                self.logger.error(message)
                sys.stderr.write('Error: ' + message)
                sys.exit()

            elif level == 4:
                self.logger.critical(message)
                sys.stderr.write('Critical: ' + message)
                sys.exit()

    """------   Getters   -----"""
    def get_candidate(self):
        return self._get_candidate

    def get_index(self):
        if hasattr(self, '_index'):
            return self._index
        else:
            path_index = self.get_param('path_index')
            if os.path.isfile(path_index):
                dtend = self.get_param('dtend')
                dtstart = self.get_param('dtstart')
                time_serie = self.get_param('ists')

                index_mat = data_utils.load_index(path_index, time_serie)
                index_mat = data_utils.take_interval(index_mat, dtend, dtstart)

                is_multi_index = isinstance(index_mat.index, pd.core.index.MultiIndex)

                if is_multi_index:
                    dates = len(set(index_mat.index.get_level_values(0)))
                    stocks = len(set(index_mat.index.get_level_values(1)))

                    self.message('Index has %s dates and %s stocks'
                                 % (str(dates), str(stocks)))

                self.set_params(index=index_mat)
                return index_mat
            else:
                self.logger.warning('No index here %s' % path_index)
                self.set_params(index=None)
                return None
                
    def get_param(self, param):
        assert type(param) == str, 'Must be a string'
        return getattr(self, '_' + param)
    
    def get_params(self, deep=True):  # @UnusedVariable
        out = {}
        for key in self.__dict__.keys():
                out[key] = self.__dict__[key]
        return out
    
    """------   Setters   -----"""
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, '_' + parameter, value)
        return self
