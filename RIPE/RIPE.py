# -*- coding: utf-8 -*-
"""
Created on 22 sept. 2016
@author: VMargot
"""
import copy
import operator
import functools
import collections
from collections import Counter

import numpy as np
import pandas as pd
import scipy.spatial.distance as scipy_dist
from scipy.stats import t, norm

from matplotlib import patches
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_array
from sklearn.externals.joblib import Parallel, delayed

"""
---------
Functions
---------
"""


def make_condition(rule):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.
    Parameters
    ----------
    rule : {rule type}
           A rule

    Return
    ------
    conditions_str : {str type}
                     A new string for the condition of the rule
    """
    conditions = rule.get_param('conditions').get_attr()
    cp = rule.get_param('cp')
    
    conditions_str = ''
    for i in range(cp):
        if i > 0:
            conditions_str += ' & '
        
        conditions_str += conditions[0][i]
        if conditions[2][i] == conditions[3][i]:
            conditions_str += ' = '
            conditions_str += str(conditions[2][i])
        else:
            conditions_str += ' $\in$ ['
            conditions_str += str(conditions[2][i])
            conditions_str += ', '
            conditions_str += str(conditions[3][i])
            conditions_str += ']'
    
    return conditions_str


def make_rules(feature_name, feature_index, X, y, method,
               cov_min, cov_max, yreal, ymean, ystd):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.

    Parameters
    ----------
    feature_name : {string type}
                   Name of the feature

    feature_index : {int type}
                    Columns index of the feature

    X : {array-like or discretized matrix, shape = [n, d]}
        The training input samples after discretization.

    y : {array-like, shape = [n]}
        The normalized target values (real numbers).

    method : {string type}
             The method mse_function or mse_function criterion

    cov_min : {float type such as 0 <= covmin <= 1}
              The minimal coverage of one rule

    cov_max : {float type such as 0 <= covmax <= 1}
              The maximal coverage of one rule

    yreal : {array-like, shape = [n]}
            The real target values (real numbers).

    ymean : {float type}
            The mean of y.

    ystd : {float type}
           The standard deviation of y.

    Return
    ------
    rules_list : {list type}
               the list of all suitable rules on the chosen feature.
    """
    xcol = X[:, feature_index]
    
    try:
        xcol = np.array(xcol, dtype=np.float64)
        notnan_vect = np.extract(np.isfinite(xcol), xcol)
        values = map(float, np.sort(list(set(notnan_vect))))
    except ValueError:
        xcol = np.array(xcol, dtype=np.str)
        values = map(str, np.sort(list(set(xcol))))
    
    rules_list = []
    for bmin in values:
        j = values.index(bmin)
        if xcol.dtype != np.str:
            for bmax in values[j:]:
                conditions = RuleConditions(features_name=[feature_name],
                                            features_index=[feature_index],
                                            bmin=[bmin],
                                            bmax=[bmax],
                                            xmax=[max(values)],
                                            xmin=[min(values)],
                                            values=values)
                
                rule = Rule(conditions)
                rules_list.append(eval_rule(rule, X, y, method, cov_min,
                                            cov_max, yreal, ymean, ystd))
        
        else:
            bmax = bmin
            conditions = RuleConditions(features_name=[feature_name],
                                        features_index=[feature_index],
                                        bmin=[bmin],
                                        bmax=[bmax],
                                        xmax=[max(values)],
                                        xmin=[min(values)],
                                        values=values)
            
            rule = Rule(conditions)
            rules_list.append(eval_rule(rule, X, y, method, cov_min,
                                        cov_max, yreal, ymean, ystd))
    
    rules_list = filter(None, rules_list)
    return rules_list


def eval_rule(rule, X, y, method, cov_min,
              cov_max, yreal, ymean, ystd):
    """
    Calculation of all statistics of an rules

    Parameters
    ----------
    rule : {rule type}
             An rule object (it means with condition on X)

    X : {array-like or discretized matrix, shape = [n, d]}
        The training input samples after discretization.

    y : {array-like, shape = [n]}
        The normalized target values (real numbers).

    method : {string type}
             The methode mse_function or mse_function criterion

    cov_min : {float type such as 0 <= covmin <= 1}
              The maximal coverage of one rule

    cov_max : {float type such as 0 <= covmax <= 1}
              The maximal coverage of one rule

    yreal : {array-like, shape = [n]}
            The real target values (real numbers).

    ymean : {float type}
            The mean of y.

    ystd : {float type}
           The standard deviation of y.


    Return
    ------
    None : if the rule does not verified criteria

    rule : {rule type}
             rule with all statistics calculated

    """
    rule.calc_stats(x=X, y=y, method=method, cov_min=cov_min,
                    cov_max=cov_max, yreal=yreal, ymean=ymean,
                    ystd=ystd)
    
    if rule.get_param('out') is False:
        return rule
    else:
        return None


def find_upcp(rule, ruleset_cp1, cp):
    """
    Calculation of all statistics of an rules

    Parameters
    ----------
    rule : {rule type}
             An rule object

    ruleset_cp1 : {ruleset type}
                 A set of rule of complexity 1

    cp : {int type, cp > 1}
         A given complexity

    Return
    ------
    rules_list : {list type}
                 List of rule made by intersection of rule with
                 rules from the rules set ruleset_cp1.

    """
    if cp == 2:
        i = ruleset_cp1.rules.index(rule)
        rules_list = map(lambda rules_cp1: rule.intersect(rules_cp1, cp),
                         ruleset_cp1[i + 1:])
        return rules_list
    
    else:
        rules_list = map(lambda rules_cp1: rule.intersect(rules_cp1, cp),
                         ruleset_cp1)
        return rules_list


def union_test(ruleset, rule, j, inter_max):
    """
    Test to add a new rule (rule) to a set of rule
    (ruleset)

    Parameters
    ----------
    ruleset : {ruleset type}
             An rule object

    rule : {rule type}
             A set of rule of complexity 1

    j : {int type or None}
        If j is not not we drop the j-th rule of ruleset
        to try to add the new rule

    inter_max : {float type, 0 <= inter_max <= 1}
                Maximal rate of intersection

    Return
    ------
    ruleset_copy : {ruleset type}
                  A set of rules with a new rule if the
                  the intersection test is satisfied

    None : If the intersection test between the new rule
           and the set of rule is not satisfied

    """
    ruleset_copy = copy.deepcopy(ruleset)
    if j is not None:
        ruleset_copy.pop(j)
        if len(ruleset_copy) > 1:
            for i in range(len(ruleset_copy)):
                rules = ruleset_copy[i]
                utest = rule.union_test(rules.get_activation(), inter_max)
                if utest is False:
                    return None
        
        if rule.union_test(ruleset_copy.calc_activation(), inter_max):
            ruleset_copy.insert(j, rule)
            return ruleset_copy
        else:
            return None
    else:
        if rule.union_test(ruleset_copy.calc_activation(), inter_max):
            ruleset_copy.append(rule)
            return ruleset_copy
        else:
            return None


def calc_ruleset_crit(ruleset, yapp, yreal, ymean, ystd, method):
    """
    Calculation of the criterium of a set of rule

    Parameters
    ----------
    ruleset : {ruleset type}
             A set of rules

    yapp : {array-like, shape = [n]}
           The normalized target values (real numbers).

    yreal : {array-like, shape = [n]}
            The real target values (real numbers).

    ymean : {float type}
            The mean of y.

    ystd : {float type}
           The standard deviation of y.

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    crit : {float type}
           The value of the criteria for the method
    """
    pred_vect = ruleset.calc_pred(y_app=yapp)
    crit = calc_crit(pred_vect, yreal, ymean, ystd, method)
    return crit


def get_variables_count(ruleset):
    """
    Get a counter of all different features in the ruleset

    Parameters
    ----------
    ruleset : {ruleset type}
             A set of rules

    Return
    ------
    count : {Counter type}
            Counter of all different features in the ruleset
    """
    col_varuleset = map(lambda rg: rg.conditions.get_param('features_name'),
                        ruleset)
    varuleset_list = functools.reduce(operator.add, col_varuleset)
    count = Counter(varuleset_list)
    
    count = count.most_common()
    return count


def dist(u, v):
    """
    Compute the distance between two prediction vector

    Parameters
    ----------
    u,v : {array type}
          A predictor vector. It means a sparse array with two
          different values 0, if the rule is not active
          and the prediction is the rule is active.

    Return
    ------
    Distance between u and v
    """
    assert len(u) == len(v), \
        'The two array must have the same length'
    u = np.sign(u)
    v = np.sign(v)
    num = np.dot(u, v)
    deno = min(np.dot(u, u),
               np.dot(v, v))
    return 1 - num / deno


def mse_function(pred_vect, y):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    pred_vect : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    crit : {float type}
           the mean squared error
    """
    assert len(pred_vect) == len(y), \
        'The two array must have the same length'
    error_vect = pred_vect - y
    crit = np.nanmean(error_vect ** 2)
    return crit


def mae_function(pred_vect, y):
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    pred_vect : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    crit : {float type}
           the mean absolute error
    """
    assert len(pred_vect) == len(y), \
        'The two array must have the same length'
    error_vect = np.abs(pred_vect - y)
    crit = np.nanmean(error_vect)
    return crit


def mape_function(pred_vect, y):
    """
    Compute the mean absolute percentage error
    "$ \\dfrac{100}{n} \\Sigma_{i=1}^{n} | \\dfrac{\\hat{y}_i - y_i}{y_i} |$"

    Parameters
    ----------
    pred_vect : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    crit : {float type}
           the mean squared error
    """
    assert len(pred_vect) == len(y), \
        'The two array must have the same length'
    error_vect = pred_vect - y
    error_vect /= y
    
    crit = np.nanmean(np.abs(error_vect)) * 100
    return crit


def calc_maximal_bend(vect):
    f_prime = np.diff(vect)
    f_second = np.diff(f_prime)
    nb_pts = len(f_second)
    
    with np.errstate(divide='ignore'):
        bend = map(lambda x: pow(1 + f_prime[x + 1] ** 2, 3. / 2) / f_second[x],
                   range(nb_pts))
    
    is_finite = np.isfinite(bend)
    bend = map(lambda c, b: abs(c) if b else 0, bend, is_finite)
    return np.argmax(bend)


def calc_crit(pred_vect, y,
              ymean=0, ystd=1,
              method='mse_function'):
    """
    Compute the criteria

    Parameters
    ----------
    pred_vect : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    ymean : {float type}
            The mean of y.

    ystd : {float type}
           The standard deviation of y.

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    crit : {float type}
           Criteria value
    """
    
    pred_vect *= ystd
    pred_vect += ymean
    y_fillna = np.nan_to_num(y)
    
    # pred_vect = np.extract(y_fillna != 0, pred_vect)
    # y_fillna = np.extract(y_fillna != 0, y_fillna)
    
    if method == 'mse_function':
        crit = mse_function(pred_vect, y_fillna)
    
    elif method == 'mae_function':
        crit = mae_function(pred_vect, y_fillna)
    
    else:
        raise 'Method %s unknown' % method
    
    return crit


def signi_test(rg, ymean, sigma, beta):
    """
    Parameters
    ----------
    rg : {Rule type}
         A rule.

    ymean : {float type}
            The mean of y.

    sigma : {float type}
            The noise estimator.
    
    beta : {float type}
            The beta factor.
            
    Return
    ------
    The bound for the conditional expectation to be significant
    """
    return beta*abs(rg.get_param('pred') - ymean) >= np.sqrt(rg.get_param('var') - sigma)


def calc_zscore(active_vect, y, th):
    """
    Compute the zscore test

    Parameters
    ----------
    active_vect : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    th : {float type}
         The threshold for the 1-type error

    Return
    ------
    The bound for the conditional expectation to be significant
    """
    nb_activation = np.sum(np.extract(active_vect, np.isfinite(y)))
    num = np.sqrt(nb_activation)
    deno = np.nanstd(y)
    ratio = deno / num
    eps = 1 - th / 2.0
    thresold = norm.ppf(eps)
    return thresold * ratio


def calc_tscore(active_vect, y, th):
    """
    Compute the tscore test

    Parameters
    ----------
    active_vect : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    th : {float type}
         The threshold for the 1-type error

    Return
    ------
    The bound for the conditional expectation to be significant
    """
    nb_activation = np.sum(np.extract(active_vect, np.isfinite(y)))
    num = np.sqrt(nb_activation)
    deno = np.nanstd(y)
    ratio = deno / num
    eps = 1 - th / 2.0
    thresold = t.ppf(eps, len(y))
    return thresold * ratio


def calc_hoeffding(active_vect, y, th):
    """
    Compute the Hoeffding test

    Parameters
    ----------
    active_vect : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    th : {float type}
         The threshold for the 1-type error

    Return
    ------
    The bound for the conditional expectation to be significant
    """
    sub_y = np.extract(active_vect, y)
    y_max = np.nanmax(sub_y)
    y_min = np.nanmin(sub_y)
    n = np.sum(active_vect)
    
    num = (y_max - y_min) * np.sqrt(np.log(2. / th))
    deno = np.sqrt(2 * n)
    return num / deno


def calc_bernstein(active_vect, y, th):
    """
    Compute the Bernstein test

    Parameters
    ----------
    active_vect : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    th : {float type}
         The threshold for the 1-type error

    Return
    ------
    The bound for the conditional expectation to be significant
    """
    sub_y = np.extract(active_vect, y)
    y_max = np.nanmax(sub_y)
    v = np.nansum(sub_y ** 2)
    n = np.sum(active_vect)
    
    val1 = y_max * np.log(2. / th)
    val2 = 72.0 * v * np.log(2. / th)
    return 1. / (6. * n) * (val1 + np.sqrt(val1 ** 2 + val2))


def calc_coverage(vect):
    """
    Compute the coverage rate of an activation vector

    Parameters
    ----------
    vect : {array type}
           A activation vector. It means a sparse array with two
           different values 0, if the rule is not active
           and the 1 is the rule is active.

    Return
    ------
    cov : {float type}
          The coverage rate
    """
    u = np.sign(vect)
    return np.dot(u, u) / float(u.size)


def calc_prediction(active_vect, y):
    """
    Compute the empirical conditional expectation of y
    knowing X

    Parameters
    ----------
    active_vect : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    Return
    ------
    pred : {float type}
           The empirical conditional expectation of y
           knowing X
    """
    y_cond = np.extract(active_vect != 0, y)
    if sum(~np.isnan(y_cond)) == 0:
        return 0
    else:
        pred = np.nanmean(y_cond)
        return pred


def calc_variance(active_vect, y):
    """
    Compute the empirical conditional expectation of y
    knowing X

    Parameters
    ----------
    active_vect : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    Return
    ------
    cond_var : {float type}
               The empirical conditional variance of y
               knowing X
    """
    # cov = calc_coverage(active_vect)
    # y_cond = active_vect * y
    # cond_var = 1. / cov * (np.mean(y_cond ** 2) - 1. / cov * np.mean(y_cond) ** 2)
    sub_y = np.extract(active_vect, y)
    cond_var = np.var(sub_y)

    return cond_var


def find_bins(xcol, nb_bucket):
    """
    Function used to find the bins to discretize xcol in nb_bucket modalities

    Parameters
    ----------
    xcol : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                Number of modalities

    Return
    ------
    bins : {ndarray type}
           The bins for disretization (result from numpy percentile function)
    """
    # Find the bins for nb_bucket
    q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
    bins = np.array([np.nanpercentile(xcol, i) for i in q_list])
    
    if bins.min() != 0:
        test_bins = bins / bins.min()
    else:
        test_bins = bins
    
    # Test if we have same bins...
    while len(set(test_bins.round(5))) != len(bins):
        # Try to decrease the number of bucket to have unique bins
        nb_bucket -= 1
        q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
        bins = np.array([np.nanpercentile(xcol, i) for i in q_list])
        if bins.min() != 0:
            test_bins = bins / bins.min()
        else:
            test_bins = bins
    
    return bins


def discretize(xcol, nb_bucket, bins=None):
    """
    Function used to have discretize xcol in nb_bucket values
    if xcol is a real series and do nothing if xcol is a string series

    Parameters
    ----------
    xcol : {Series type}
           Series to discretize

    nb_bucket : {int type}
                Number of modalities

    bins : {ndarray type}, optional, default None
           If you have already calculate the bins for xcol

    Return
    ------
    xcol_discretized : {Series type}
                       The discretization of xcol
    """
    if xcol.dtype.type != np.object_:
        # extraction of the list of xcol values
        notnan_vect = np.extract(np.isfinite(xcol), xcol)
        nan_index = ~np.isfinite(xcol)
        # Test if xcol have more than nb_bucket different values
        if len(set(notnan_vect)) >= nb_bucket or bins is not None:
            if bins is None:
                bins = find_bins(xcol, nb_bucket)
            # discretization of the xcol with bins
            xcol_discretized = np.digitize(xcol, bins=bins)
            xcol_discretized = np.array(xcol_discretized, dtype='float')
            
            if sum(nan_index) > 0:
                xcol_discretized[nan_index] = np.nan
            
            return xcol_discretized
        
        return xcol
    
    else:
        return xcol


class RuleConditions(object):
    """
    Class for binary rule condition
    """
    
    def __init__(self, features_name, features_index,
                 bmin, bmax, xmin, xmax, values=list([])):
        
        assert isinstance(features_name, collections.Iterable), \
            'Type of parameter must be iterable' % features_name
        self.features_name = features_name
        cp = len(features_name)
        
        assert isinstance(features_index, collections.Iterable), \
            'Type of parameter must be iterable' % features_name
        assert len(features_index) == cp, \
            'Parameters must have the same length' % features_name
        self.features_index = features_index
        
        assert isinstance(bmin, collections.Iterable), \
            'Type of parameter must be iterable' % features_name
        assert len(bmin) == cp, \
            'Parameters must have the same length' % features_name
        assert isinstance(bmax, collections.Iterable), \
            'Type of parameter must be iterable' % features_name
        assert len(bmax) == cp, \
            'Parameters must have the same length' % features_name
        if type(bmin[0]) != np.string_:
            assert all(map(lambda a, b: a <= b, bmin, bmax)), \
                'Bmin must be smaller or equal than bmax (%s)' \
                % features_name
        self.bmin = bmin
        self.bmax = bmax
        
        assert isinstance(xmax, collections.Iterable), \
            'Type of parametre must be iterable' % features_name
        assert len(xmax) == cp, \
            'Parameters must have the same length' % features_name
        assert isinstance(xmin, collections.Iterable), \
            'Type of parameter must be iterable' % features_name
        assert len(xmin) == cp, \
            'Parameters must have the same length' % features_name
        self.xmin = xmin
        self.xmax = xmax
        
        self.values = [values]
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        features = self.features_name
        return "Var: %s, Bmin: %s, Bmax: %s" % (features, self.bmin, self.bmax)
    
    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
    
    def __hash__(self):
        to_hash = [(self.features_index[i], self.features_name[i],
                    self.bmin[i], self.bmax[i])
                   for i in range(len(self.features_index))]
        to_hash = frozenset(to_hash)
        return hash(to_hash)
    
    def transform(self, xmat):
        """
        Transform a matrix xmat into an activation vector.
        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise.

        Parameters
        ----------
        xmat: {array-like matrix, shape=(n_samples, n_features)}
              Input data

        Returns
        -------
        active_vect: {array-like matrix, shape=(n_samples, 1)}
                     The activation vector
        """
        cp = len(self.features_name)
        geq_min = True
        leq_min = True
        not_nan = True
        for i in range(cp):
            col_index = self.features_index[i]
            x_col = xmat[:, col_index]
            
            # Turn x_col to array
            x_col = np.squeeze(np.asarray(x_col))
            
            if type(self.bmin[i]) == str:
                x_col = np.array(x_col, dtype=np.str)
                
                temp = (x_col == self.bmin[i])
                temp |= (x_col == self.bmax[i])
                geq_min &= temp
                leq_min &= True
                not_nan &= True
            else:
                x_col = np.array(x_col, dtype=np.float)
                
                x_temp = map(lambda x: self.bmin[i] - 1 if x != x else x,
                             x_col)
                geq_min &= np.greater_equal(x_temp, self.bmin[i])
                
                x_temp = map(lambda x: self.bmax[i] + 1 if x != x else x,
                             x_col)
                leq_min &= np.less_equal(x_temp, self.bmax[i])
                
                not_nan &= np.isfinite(x_col)
        
        active_vect = 1 * (geq_min & leq_min & not_nan)
        
        return active_vect
    
    """------   Getters   -----"""
    
    def get_param(self, param):
        """
        To get the parameter param
        """
        assert type(param) == str, \
            'Must be a string'
        
        return getattr(self, param)
    
    def get_attr(self):
        """
        To get a list of attributes of self.
        It is useful to quickly create a RuleConditions
        from intersection of two rules
        """
        return [self.features_name,
                self.features_index,
                self.bmin, self.bmax,
                self.xmin, self.xmax]
    
    """------   Setters   -----"""
    
    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
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
        self.cp = len(rule_conditions.get_param('features_index'))
    
    def __repr__(self):
        return self.__str__()
    
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
        return 'rule: ' + self.conditions.__str__()
    
    def __hash__(self):
        return hash(self.conditions)
    
    def test_included(self, rule):
        """
        Test to know if a rule (self) and an other (rule)
        are included
        """
        activation_self = self.get_activation()
        activation_other = rule.get_activation()
        
        intersection = np.logical_and(activation_self, activation_other)
        
        if (np.allclose(intersection, activation_self) or
                np.allclose(intersection, activation_other)):
            return None
        else:
            return 1 * intersection
    
    def test_variables(self, rule):
        """
        Test to know if a rule (self) and an other (rule)
        have conditions on the same features.
        """
        c1 = self.conditions
        c2 = rule.conditions
        
        c1_name = c1.get_param('features_name')
        c2_name = c2.get_param('features_name')
        if len(set(c1_name).intersection(c2_name)) != 0:
            return True
        else:
            return False
    
    def test_cp(self, rule, cp):
        """
        Test to know if a rule (self) and an other (rule)
        could be intersected to have a new rule of complexity cp.
        """
        return self.get_param('cp') + rule.get_param('cp') == cp
    
    def intersect_test(self, rule, cp):
        """
        Test to know if a rule (self) and an other (rule)
        could be intersected.

        Test 1: the sum of complexities of self and rule are egal to cp
        Test 2: self and rule have not condition on the same variable
        Test 3: self and rule have not included activation
        """
        if self.test_cp(rule, cp):
            if self.test_variables(rule) is False:
                return self.test_included(rule=rule)
            else:
                return None
        else:
            return None
    
    def union_test(self, activation, inter_max=0.80):
        """
        Test to know if a rule (self) and an activation vector have
        at more inter_max percent of points in common
        """
        self_vect = self.get_activation()
        intersect_vect = np.logical_and(self_vect, activation)
        
        pts_inter = np.sum(intersect_vect)
        pts_rule = np.sum(activation)
        pts_self = np.sum(self_vect)
        
        ans = ((pts_inter < inter_max * pts_self) and
               (pts_inter < inter_max * pts_rule))
        
        return ans
    
    def intersect_conditions(self, rule):
        """
        Compute an RuleCondition object from the intersection of an rule
        (self) and an other (rulessert)
        """
        conditions_1 = self.conditions
        conditions_2 = rule.conditions
        
        conditions = map(lambda c1, c2: c1 + c2, conditions_1.get_attr(),
                         conditions_2.get_attr())
        
        return conditions
    
    def intersect(self, rule, cp):
        """
        Compute a suitable rule object from the intersection of an rule
        (self) and an other (rulessert).
        Suitable means that self and rule satisfied the intersection test
        """
        if self.get_param('pred') * rule.get_param('pred') > 0:
            
            activation = self.intersect_test(rule, cp)
            if activation is not None:
                conditions_list = self.intersect_conditions(rule)
                
                new_conditions = RuleConditions(features_name=conditions_list[0],
                                                features_index=conditions_list[1],
                                                bmin=conditions_list[2],
                                                bmax=conditions_list[3],
                                                xmax=conditions_list[5],
                                                xmin=conditions_list[4])
                new_rule = Rule(new_conditions)
                new_rule.set_params(activation=activation)
                return new_rule
            else:
                return None
        else:
            return None
    
    def calc_stats(self, x, y, method='mse_function',
                   cov_min=0.01, cov_max=0.5, yreal=None,
                   ymean=0, ystd=1):
        """
        Calculation of all statistics of an rules

        Parameters
        ----------
        x : {array-like or discretized matrix, shape = [n, d]}
            The training input samples after discretization.

        y : {array-like, shape = [n]}
            The normalized target values (real numbers).

        method : {string type}
                 The method mse_function or mse_function criterion

        cov_min : {float type such as 0 <= covmin <= 1}, default 0.5
                  The minimal coverage of one rule

        cov_max : {float type such as 0 <= covmax <= 1}, default 0.5
                  The maximal coverage of one rule

        yreal : {array-like, shape = [n]}, default None
                The real target values (real numbers).

        ymean : {float type}, default 0
                The mean of y.

        ystd : {float type}, default 1
               The standard deviation of y.

        Return
        ------
        None : if the rule does not verified coverage conditions
        """
        self.set_params(out=False)
        active_vect = self.calc_activation(x=x)
        
        self.set_params(activation=active_vect)
        
        cov = calc_coverage(active_vect)
        self.set_params(cov=cov)
        
        if cov > cov_max or cov < cov_min:
            self.set_params(out=True)
            self.set_params(reason='Cov')
            return
        
        else:
            pred = calc_prediction(active_vect, y)
            self.set_params(pred=pred)

            cond_var = calc_variance(active_vect, y)
            self.set_params(var=cond_var)
            
            if yreal is None:
                yreal = y
            
            pred_vect = active_vect * pred
            cplt_val = calc_prediction(1 - active_vect, y)
            np.place(pred_vect, pred_vect == 0, cplt_val)
            
            rez = calc_crit(pred_vect, yreal, ymean, ystd, method)
            self.set_params(crit=rez)
    
    def calc_activation(self, x=None):
        """
        Compute the activation vector of an rule
        """
        return self.conditions.transform(x)
    
    def predict(self, x=None):
        """
        Compute the prediction of an rule
        """
        pred = self.get_param('pred')
        if x is not None:
            activation = self.calc_activation(x=x)
        else:
            activation = self.get_activation()
        
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
        
        y = np.extract(np.isfinite(y), y)
        pred_vect = np.extract(np.isfinite(y), pred_vect)
        
        if score_type == 'Classification':
            th_val = (min(y) + max(y)) / 2.0
            pred_vect = np.array(map(lambda p: min(y) if p < th_val else max(y), pred_vect))
            return accuracy_score(y, pred_vect)
        
        elif score_type == 'Regression':
            return r2_score(y, pred_vect, sample_weight=sample_weight,
                            multioutput='variance_weighted')
    
    def make_name(self, num, learning=None):
        """
        Add an attribute name to self

        Parameters
        ----------
        num : int
              index of the rule in an ruleset

        learning : Learning object, default None
                   If leaning is not None the name of self will
                   be defined with the name of learning
        """
        name = 'R ' + str(num)
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
        """
        To get the parameter param
        """
        assert type(param) == str, 'Must be a string'
        assert hasattr(self, param), \
            'self.%s must be calculate before' % param
        return getattr(self, param)
    
    def get_activation(self):
        """
        To get the activation vector of self.
        If it does not exist the function return None
        """
        if hasattr(self, 'activation'):
            return self.get_param('activation')
        else:
            return None
    
    def get_pred_vect(self):
        """
        To get the activation vector of self.
        If it does not exist the function return None
        """
        if hasattr(self, 'pred'):
            pred = self.get_param('pred')
            if hasattr(self, 'activation'):
                return pred * self.get_param('activation')
            else:
                return None
        else:
            return None
    
    """------   Setters   -----"""
    
    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class RuleSet(object):
    """
    Class for a ruleset. It's a kind of list of rule object
    """
    
    def __init__(self, rs):
        if type(rs) == list:
            self.rules = rs
        elif type(rs) == RuleSet:
            self.rules = rs.get_rules()
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return 'ruleset: %s rules' % str(len(self.rules))
    
    def __gt__(self, val):
        return map(lambda rg: rg > val, self.rules)
    
    def __lt__(self, val):
        return map(lambda rg: rg < val, self.rules)
    
    def __ge__(self, val):
        return map(lambda rg: rg >= val, self.rules)
    
    def __le__(self, val):
        return map(lambda rg: rg <= val, self.rules)
    
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
    
    def __delitem__(self, rules_id):
        del self.rules[rules_id]
    
    def append(self, rule):
        """
        Add one rule to a RuleSet object (self).
        """
        assert rule.__class__ == Rule, 'Must be a rule object (try extend)'
        self.rules.append(rule)
    
    def extend(self, ruleset):
        """
        Add rules form a ruleset to a RuleSet object (self).
        """
        assert ruleset.__class__ == RuleSet, 'Must be a ruleset object'
        'ruleset must have the same Learning object'
        rules_list = ruleset.get_rules()
        self.rules.extend(rules_list)
        return self
    
    def insert(self, idx, rule):
        """
        Insert one rule to a RuleSet object (self) at the position idx.
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.rules.insert(idx, rule)
    
    def pop(self, idx=None):
        """
        Drop the rule at the position idx.
        """
        self.rules.pop(idx)
    
    def extract_greater(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        greater than val.
        """
        rules_list = filter(lambda rg: rg.get_param(param) > val, self)
        return RuleSet(rules_list)
    
    def extract_least(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        least than val.
        """
        rules_list = filter(lambda rg: rg.get_param(param) < val, self)
        return RuleSet(rules_list)
    
    def extract_cp(self, cp):
        """
        Extract a RuleSet object from self such as each rules have a
        complexity cp.
        """
        rules_list = filter(lambda rg: rg.get_param('cp') == cp, self)
        return RuleSet(rules_list)
    
    def extract(self, param, val):
        """
        Extract a RuleSet object from self such as each rules have a param
        equal to val.
        """
        rules_list = filter(lambda rg: rg.get_param(param) == val, self)
        return RuleSet(rules_list)
    
    def index(self, rule):
        """
        Get the index a rule in a RuleSet object (self).
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.get_rules().index(rule)
    
    def replace(self, idx, rule):
        """
        Replace rule at position idx in a RuleSet object (self)
        by a new rule.
        """
        self.rules.pop(idx)
        self.rules.insert(idx, rule)
    
    def sort_by(self, crit, maximized):
        """
        Sort the RuleSet object (self) by a criteria crit
        """
        self.rules.sort(key=lambda x: x.get_param(crit),
                        reverse=maximized)
    
    def drop_duplicates(self):
        """
        Drop duplicates rules in RuleSet object (self)
        """
        rules_list = list(set(self.rules))
        return RuleSet(rules_list)
    
    def to_df(self, cols=None):
        """
        To transform an ruleset into a pandas DataFrame
        """
        if cols is None:
            cols = ['Features_Name', 'BMin', 'BMax',
                    'Cov', 'Pred', 'Var', 'Crit']
        
        df = pd.DataFrame(index=self.get_rules_name(),
                          columns=cols)
        
        for col_name in cols:
            att_name = col_name.lower()
            if all(map(lambda rg: hasattr(rg, att_name),
                       self)):
                df[col_name] = map(lambda rg:
                                   rg.get_param(att_name),
                                   self)
            
            elif all(map(lambda rg: hasattr(rg.conditions, att_name.lower()),
                         self)):
                df[col_name] = map(lambda rg:
                                   rg.conditions.get_param(att_name),
                                   self)
        
        return df
    
    def calc_pred(self, y_app, x=None):
        """
        Computes the prediction vector
        using an rule based partition
        """
        # Activation of all rules in the learning set
        activ_mat = np.matrix(map(lambda rules: rules.activation, self))
        
        if x is None:
            pred_mat = activ_mat.T
        else:
            pred_mat = map(lambda rules: rules.calc_activation(x), self)
            pred_mat = np.matrix(pred_mat).T
        
        nopred_mat = np.logical_not(pred_mat)
        
        nb_rules_active = pred_mat.sum(axis=1)
        
        # Activation of the intersection of all activated rules at each row
        dot_activation = np.dot(pred_mat, activ_mat)
        # Activation vectors for intersection of activated rules
        dot_activation = np.matrix(dot_activation == nb_rules_active,
                                   dtype='int')
        
        # Activation of the intersection of all NOT activated rules at each row
        dot_noactivation = np.dot(nopred_mat, activ_mat)
        dot_noactivation = np.matrix(dot_noactivation,
                                     dtype='int')
        
        # Calculation of the binary vector for cells of the partition et each row
        cells = ((dot_activation - dot_noactivation) > 0)
        
        # Calculation of the conditional expectation in each cell
        pred_vect = map(lambda act: calc_prediction(act, y_app),
                        cells)
        
        pred_vect = np.array(pred_vect)
        
        return pred_vect
    
    def calc_activation(self):
        """
        Compute the  activation vector of a set of rules
        """
        active_vect = map(lambda rg: rg.get_activation(), self)
        active_vect = sum(active_vect)
        active_vect = 1 * active_vect.astype('bool')
        
        return active_vect
    
    def calc_coverage(self):
        """
        Compute the coverage rate of a set of rules
        """
        if len(self) > 0:
            active_vect = self.calc_activation()
            cov = calc_coverage(active_vect)
        else:
            cov = 0.0
        return cov
    
    def predict(self, y_app, x, ymean, ystd):
        """
        Computes the prediction vector for a given X and a given aggregation method
        """
        pred_vect = self.calc_pred(y_app, x)
        return pred_vect * ystd + ymean
    
    def make_rule_names(self):
        """
        Add an attribute name at each rule of self
        """
        map(lambda rule, rules_id: rule.make_name(rules_id),
            self, range(len(self)))
    
    """------   Getters   -----"""
    
    def get_rules_param(self, param):
        """
        To get the list of a parameter param of the rules in self
        """
        return map(lambda rule: rule.get_param(param), self)
    
    def get_rules_name(self):
        """
        To get the list of the name of rules in self
        """
        try:
            return self.get_rules_param('name')
        except AssertionError:
            self.make_rule_names()
            return self.get_rules_param('name')
    
    def get_rules(self):
        """
        To get the list of rule in self
        """
        return self.rules
    
    """------   Setters   -----"""
    
    def set_rules(self, rules_list):
        """
        To set a list of rule in self
        """
        assert type(rules_list) == list, 'Must be a list object'
        self.rules = rules_list


class Learning(BaseEstimator):
    """
    ...
    """
    
    def __init__(self, **parameters):
        """

        Parameters
        ----------
        maximized : {boolean type} default False
                    To choose if the criterion must be maximized
                    Otherwise it will be minimized

        method : {string type} default mse_function if y has more than
                 2 differents values
                 Choose among the mse_function and mse_function criterion

        signicrit : {string type} default bernstein if the number of row is
                   greater than 30 else tscore
                   Choose among zscore, hoeffding and bernstein

        th : {float type such as 0 < th < 1} default 0.05
             Choose the threshold for the type 1 error

        nb_bucket : {int type} default max(3, n^1/d) with n the number of row
                    and d the number of features
                    Choose the number a bucket for the discretization

        cp : {int type} default d
             Choose the maximal complexity of one rule

        covmin : {float type such as 0 < covmin < 1} default 1/(nb_bucket^d)
                 Choose the minimal coverage of one rule

        covmax : {float type such as 0 < covmax < 1} default 1/log(nb_bucket)
                 Choose the minimal coverage of one rule

        nb_candidates : {int type} default 300
                    Choose the number of candidates to increase complexity

        intermax : {float type such as 0 <= intermax <= 1} default 1
                   Choose the maximal intersection rate begin a rule and
                   a current selected ruleset

        nb_jobs : {int type} default number of core -2
                  Select the number of CPU used

        fullselection : {boolean type} default True
                        Choose if the selection is among all complexity (True)
                        or a selection by complexity (False)
        """
        self.selected_rs = RuleSet([])
        self.ruleset = RuleSet([])
        self.bins = dict()
        
        for arg, val in parameters.items():
            setattr(self, arg, val)
        
        if hasattr(self, 'th') is False:
            self.th = 0.05
        
        if hasattr(self, 'nb_jobs') is False:
            self.nb_jobs = -2
        
        if hasattr(self, 'nb_candidates') is False:
            self.nb_candidates = 300
        
        if hasattr(self, 'intermax') is False:
            self.intermax = 1.0
        
        if hasattr(self, 'fullselection') is False:
            self.fullselection = True
        
        if hasattr(self, 'maximized') is False:
            self.maximized = False
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        learning = self.get_param('cpname') + ': '
        learning += self.get_param('target')
        return learning
    
    def fit(self, X, y, features_name=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like or sparse matrix, shape = [n, d]}
            The training input samples.

        y : {array-like, shape = [n]}
            The target values (real numbers).

        features_name : {list}, optional
                        Name of each features
        """
        
        # Check type for data
        X = check_array(X, dtype=None, force_all_finite=False)  # type: np.object_
        y = check_array(y, dtype=None, ensure_2d=False,
                        force_all_finite=False)  # type: np.object_

        alpha = 1./3
        # Creation of data-driven parameters
        if hasattr(self, 'beta') is False:
            beta = 1./pow(X.shape[0], alpha/2. - 1./4)
            self.set_params(beta=beta)

        if hasattr(self, 'nb_bucket') is False:
            nb_bucket = max(5, int(np.sqrt(pow(X.shape[0],
                                               1. / X.shape[1]))))
            
            nb_bucket = min(nb_bucket, X.shape[0])
            self.set_params(nb_bucket=nb_bucket)
        
        if hasattr(self, 'covmin') is False:
            covmin = 1./pow(X.shape[0], alpha)
            self.set_params(covmin=covmin)
        
        if hasattr(self, 'covmax') is False:
            covmax = 1.0 / np.log(np.sqrt(self.get_param('nb_bucket')))
            covmax = min(0.99, covmax)
            
            self.set_params(covmax=covmax)

        if hasattr(self, 'calcmethod') is False:
            if len(set(y)) > 2:
                # Binary classification case
                calcmethod = 'mse_function'
            else:
                # Regression case
                calcmethod = 'mae_function'
            self.set_params(calcmethod=calcmethod)
        
        features_index = range(X.shape[1])
        if features_name is None:
            features_name = map(lambda i: 'X' + str(i), features_index)
        
        self.set_params(features_index=features_index)
        self.set_params(features_name=features_name)
        
        if hasattr(self, 'cp') is False:
            cp = len(features_name)
            self.set_params(cp=cp)
        
        # Turn the matrix X in a discret matrix
        X_ripe = self.discretize(X)
        self.set_params(X=X_ripe)
        
        self.set_params(yreal=y)
        
        # Normalization of y
        ymean = np.nanmean(y)
        ystd = np.nanstd(y)
        self.set_params(ymean=ymean)
        self.set_params(ystd=ystd)
        
        y_new = y - ymean
        y_new /= ystd
        self.set_params(y=y_new)

        # By independance of the noise
        if hasattr(self, 'sigma'):
            sigma = self.sigma / ystd ** 2
            self.set_params(sigma=sigma)
        
        # looking for good rules
        self.find_rules()  # works in columns not in lines
        
        self.set_params(fitted=True)
    
    def find_rules(self):
        """
        Find all rules for all complexity <= cp
        then selects the best subset by minimization
        of the empirical risk
        """
        complexity = self.get_param('cp')
        maximized = self.get_param('maximized')
        
        assert complexity > 0, \
            'Complexity must be strictly superior to 0'
        
        selected_rs = self.get_param('selected_rs')
        
        # -----------
        # DESIGN PART
        # -----------
        self.calc_cp1()
        ruleset = self.get_param('ruleset')
        
        if len(ruleset) > 0:
            for cp in range(2, complexity + 1):
                print('Design for complexity %s' % str(cp))
                if len(selected_rs.extract_cp(cp)) == 0:
                    # seeking a set of rules with a complexity cp
                    ruleset_cpup = self.up_complexity(cp)
                    
                    if len(ruleset_cpup) > 0:
                        ruleset += ruleset_cpup
                        self.set_params(ruleset=ruleset)
                    else:
                        print('No rules for complexity %s' % str(cp))
                        break
                    
                    ruleset.sort_by('crit', maximized)
            self.set_params(ruleset=ruleset)
            
            # --------------
            # SELECTION PART
            # --------------
            print('----- Selection ------')
            selection_type = self.get_param('fullselection')
            if selection_type:
                selected_rs = self.select_rules(0)
            else:
                selected_rs = self.get_param('selected_rs')
                for cp in range(1, complexity + 1):
                    selected_rs += self.select_rules(cp)
            
            ruleset.make_rule_names()
            self.set_params(ruleset=ruleset)
            selected_rs.make_rule_names()
            self.set_params(selected_rs=selected_rs)
        
        else:
            print('No rules found !')
    
    def calc_cp1(self):
        """
        Compute all rules of complexity one and keep the best.
        """
        features_name = self.get_param('features_name')
        features_index = self.get_param('features_index')
        X = self.get_param('X')
        method = self.get_param('calcmethod')
        y = self.get_param('y')
        yreal = self.get_param('yreal')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        cov_max = self.get_param('covmax')
        cov_min = self.get_param('covmin')
        
        jobs = min(len(features_name), self.get_param('nb_jobs'))
        
        if jobs == 1:
            ruleset = map(lambda var, idx: make_rules(var, idx, X, y, method,
                                                      cov_min, cov_max,
                                                      yreal, ymean, ystd),
                          features_name, features_index)
        else:
            ruleset = Parallel(n_jobs=jobs, backend="multiprocessing")(
                delayed(make_rules)(var, idx, X, y, method, cov_min,
                                    cov_max, yreal, ymean, ystd)
                for var, idx in zip(features_name, features_index))
        
        ruleset = functools.reduce(operator.add, ruleset)
        
        ruleset = RuleSet(ruleset)
        ruleset.sort_by('crit', self.get_param('maximized'))
        
        self.set_params(ruleset=ruleset)
    
    def up_complexity(self, cp):
        """
        Returns a ruleset of rules with complexity=cp.
        """
        nb_jobs = self.get_param('nb_jobs')
        X = self.get_param('X')
        method = self.get_param('calcmethod')
        y = self.get_param('y')
        yreal = self.get_param('yreal')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        cov_max = self.get_param('covmax')
        cov_min = self.get_param('covmin')
        
        rules_list = self.find_candidates(cp)
        
        if len(rules_list) > 0:
            if nb_jobs == 1:
                ruleset = map(lambda rule: eval_rule(rule, X, y, method, cov_min,
                                                     cov_max, yreal, ymean, ystd),
                              rules_list)
            else:
                ruleset = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
                    delayed(eval_rule)(rule, X, y, method, cov_min,
                                       cov_max, yreal, ymean, ystd)
                    for rule in rules_list)
            
            ruleset = filter(None, ruleset)
            ruleset_cpup = RuleSet(ruleset)
            ruleset_cpup = ruleset_cpup.drop_duplicates()
            return ruleset_cpup
        else:
            return []
    
    def select_candidates(self, rules_cp):
        """
        Returns a selection of candidates to increase complexity
        for a given complexity (cp)
        """
        ruleset = self.get_param('ruleset')
        ruleset_candidates = ruleset.extract_cp(rules_cp)
        ruleset_candidates.sort_by('var', True)
        
        nb_candidates = self.get_param('nb_candidates')
        if nb_candidates is not None:
            if len(ruleset_candidates) > nb_candidates:
                pos_ruleset = ruleset_candidates.extract_greater('pred', 0)
                neg_ruleset = ruleset_candidates.extract_least('pred', 0)
                
                id_pos = float(len(pos_ruleset)) / len(ruleset_candidates) * nb_candidates
                id_neg = float(len(neg_ruleset)) / len(ruleset_candidates) * nb_candidates
                
                rules_list = pos_ruleset[:int(id_pos)]
                rules_list += neg_ruleset[:int(id_neg)]
                
                ruleset_candidates = RuleSet(list(rules_list))
        return ruleset_candidates
    
    def find_candidates(self, cp):
        """
        Returns the intersection of all suitable rules
        for a given complexity (cp)
        """
        ruleset_cp1 = self.select_candidates(1)
        
        ruleset_candidate = self.select_candidates(cp - 1)
        
        if len(ruleset_candidate) > 0:
            rules_list = self.find_complexe_rules(cp, ruleset_cp1,
                                                  ruleset_candidate)
            
            return rules_list
        else:
            return []
    
    def find_complexe_rules(self, cp, ruleset_cp1, ruleset_candidate):
        """
        Returns a list of Rule object designing by intersection of rule from
        ruleset_cp1 and rule from ruleset_candidate
        """
        nb_jobs = self.get_param('nb_jobs')
        
        if nb_jobs == 1:
            rules_list = map(lambda rule: find_upcp(rule, ruleset_cp1, cp),
                             ruleset_candidate)
        else:
            rules_list = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
                delayed(find_upcp)(rule, ruleset_cp1, cp)
                for rule in ruleset_candidate)
        
        rules_list = functools.reduce(operator.add, rules_list)
        
        rules_list = filter(None, rules_list)  # to drop bad rules
        rules_list = list(set(rules_list))  # to drop duplicates
        return rules_list
    
    def select_rules(self, cp):
        """
        Returns a subset of a given ruleset.
        This subset minimizes the empirical contrast on the learning set
        """
        maximized = self.get_param('maximized')
        ymean = self.get_param('ymean')
        ruleset = self.get_param('ruleset')
        beta = self.get_param('beta')

        if hasattr(self, 'sigma'):
            sigma = self.get_param('sigma')
        else:
            sigma = min(ruleset.get_rules_param('var'))
            self.set_params(sigma=sigma)
        
        if cp > 0:
            sub_ruleset = ruleset.extract_cp(cp)
        else:
            sub_ruleset = copy.deepcopy(ruleset)
            
        print 'Number rules: %s' % str(len(sub_ruleset))
        sub_ruleset = RuleSet(filter(lambda rg: signi_test(rg, ymean, sigma, beta),
                                     sub_ruleset))
        print 'Number rules after significant test: %s' % str(len(sub_ruleset))
        
        if len(sub_ruleset) > 0:
            sub_ruleset.sort_by('crit', maximized)
            selected_rs = self.minimized_risk(sub_ruleset)
        else:
            selected_rs = RuleSet([])
            print('No rules selected')
            
        return selected_rs
    
    def minimized_risk(self, ruleset):
        """
        Returns a subset of a given ruleset. This subset is seeking by
        minimization/maximization of the criterion on the training set
        """
        yapp = self.get_param('y')
        yreal = self.get_param('yreal')
        ystd = self.get_param('ystd')
        ymean = self.get_param('ymean')
        method = self.get_param('calcmethod')
        maximized = self.get_param('maximized')
        inter_max = self.get_param('intermax')
        
        # win = 10

        # Then optimization
        selected_rs = RuleSet(ruleset[:1])
        old_crit = calc_ruleset_crit(selected_rs, yapp, yreal,
                                     ymean, ystd, method)
        crit_evo = [old_crit]
        nb_rules = len(ruleset)

        for i in range(1, nb_rules):
            new_rules = ruleset[i]
            iter_list = [None]
            
            if len(selected_rs) > 1:
                iter_list += range(len(selected_rs))
            
            ruleset_list = []
            for j in iter_list:
                break_loop = False
                ruleset_copy = copy.deepcopy(selected_rs)
                if j is not None:
                    ruleset_copy.pop(j)
                    if new_rules.union_test(ruleset_copy.calc_activation(),
                                            inter_max):
                        if len(ruleset_copy) > 1:
                            for rules in ruleset_copy:
                                utest = new_rules.union_test(rules.get_activation(),
                                                             inter_max)
                                if not utest:
                                    break_loop = True
                                    break
                        if break_loop:
                            continue
                        
                        ruleset_copy.insert(j, new_rules)
                        ruleset_list.append(ruleset_copy)
                    else:
                        continue
                
                else:
                    utest = map(lambda e: new_rules.union_test(e.get_activation(),
                                                               inter_max), ruleset_copy)
                    if all(utest) and new_rules.union_test(ruleset_copy.calc_activation(),
                                                           inter_max):
                        ruleset_copy.append(new_rules)
                        ruleset_list.append(ruleset_copy)
            
            if len(ruleset_list) > 0:
                crit_list = map(lambda e: calc_ruleset_crit(e, yapp, yreal,
                                                            ymean, ystd, method),
                                ruleset_list)
                
                if maximized:
                    ruleset_idx = int(np.argmax(crit_list))
                    if crit_list[ruleset_idx] >= old_crit:
                        selected_rs = copy.deepcopy(ruleset_list[ruleset_idx])
                        old_crit = crit_list[ruleset_idx]
                else:
                    ruleset_idx = int(np.argmin(crit_list))
                    if crit_list[ruleset_idx] <= old_crit:
                        selected_rs = copy.deepcopy(ruleset_list[ruleset_idx])
                        old_crit = crit_list[ruleset_idx]
            
            crit_evo.append(old_crit)
            
            # Stopping Criteria
            # if len(crit_evo) > win:
            #     nb_zeros = list(np.diff(crit_evo[-win:])).count(0)
            #     if nb_zeros == win-1:
            #         break
        
        self.set_params(critlist=crit_evo)
        return selected_rs
    
    def predict(self, X, check_input=True):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        application of the selected ruleset on X.

        Parameters
        ----------
        X : {array type or sparse matrix of shape = [n_samples, n_features]}
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a spares matrix is provided, it will be
            converted into a spares ``csr_matrix``.

        check_input : bool type

        Returns
        -------
        y : {array type of shape = [n_samples]}
            The predicted values.
        """
        y_app = self.get_param('y')
        
        X = self.validate_X_predict(X, check_input)
        x_copy = self.discretize(X)
        
        ruleset = self.get_param('selected_rs')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        
        pred_vect = ruleset.predict(y_app, x_copy, ymean, ystd)
        pred_vect = np.array(map(lambda p: ymean if p != p else p, pred_vect))
        
        return pred_vect
    
    def score(self, x, y, sample_weight=None):
        """
        Returns the coefficient of determination R^2 of the prediction
        if y is continuous. Else if y in {0,1} then Returns the mean
        accuracy on the given test data and labels {0,1}.

        Parameters
        ----------
        x : {array type or sparse matrix of shape = [n_samples, n_features]}
            Test samples.

        y : {array type of shape = [n_samples]}
            True values for y.

        sample_weight : {array type of shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y in R.
            or
        score : float
            Mean accuracy of self.predict(X) wrt. y in {0,1}
        """
        x_copy = copy.copy(x)
        
        pred_vect = self.predict(x_copy)
        pred_vect = np.nan_to_num(pred_vect)
        
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
    
    """------   Data functions   -----"""
    
    def validate_X_predict(self, X, check_input):
        """
        Validate X whenever one tries to predict, apply, predict_proba
        """
        if hasattr(self, 'fitted') is False:
            raise AttributeError("Estimator not fitted, "
                                 "call 'fit' before exploiting the model.")
        
        if check_input:
            X = check_array(X, dtype=None, force_all_finite=False)  # type: np.object_
            
            n_features = X.shape[1]
            input_features = self.get_param('features_name')
            if len(input_features) != n_features:
                raise ValueError("Number of features of the model must "
                                 "match the input. Model n_features is %s and "
                                 "input n_features is %s "
                                 % (input_features, n_features))
        
        return X
    
    def discretize(self, x):
        """
        Used to have discrete values for each series
        to avoid float

        Parameters
        ----------
        x : {array, matrix type}, shape=[n_samples, n_features]
            Features matrix

        Return
        -------
        col : {array, matrix type}, shape=[n_samples, n_features]
              Features matrix with each features values discretized
              in nb_bucket values
        """
        nb_col = x.shape[1]
        nb_bucket = self.get_param('nb_bucket')
        bins_dict = self.get_param('bins')
        features_name = self.get_param('features_name')
        
        x_mat = []
        for i in range(nb_col):
            x_col = x[:, i]
            try:
                x_col = np.array(x_col.flat, dtype=np.float64)
            except ValueError:
                x_col = np.array(x_col.flat, dtype=np.str)
            
            var_name = features_name[i]
            
            if x_col.dtype.type != np.string_:
                if var_name not in bins_dict:
                    if len(set(x_col)) >= nb_bucket:
                        bins = find_bins(x_col, nb_bucket)
                        col = discretize(x_col, nb_bucket, bins)
                        bins_dict[var_name] = bins
                    else:
                        col = x_col
                else:
                    bins = bins_dict[var_name]
                    col = discretize(x_col, nb_bucket, bins)
            else:
                col = x_col
            
            x_mat.append(col)
        
        return np.array(x_mat).T
    
    def plot_rules(self, var1, var2, cp=None,
                   col_pos='red', col_neg='blue'):
        """
        Plot the rectangle activation zone of rules in a 2D plot
        the color is corresponding to the intensity of the prediction

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
        nb_bucket = self.get_param('nb_bucket')
        
        if cp is not None:
            sub_ruleset = selected_rs.extract_cp(cp)
        else:
            sub_ruleset = selected_rs
        
        plt.plot()
        
        for rg in sub_ruleset:
            rg_condition = rg.conditions
            
            var = rg_condition.get_param('features_index')
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
                if var[0] == var1:
                    p = patches.Rectangle((bmin[0], 0),  # origin
                                          (bmax[0] - bmin[0]) + 0.99,  # width
                                          nb_bucket,  # height
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)
                
                elif var[0] == var2:
                    p = patches.Rectangle((0, bmin[0]),
                                          nb_bucket,
                                          (bmax[0] - bmin[0]) + 0.99,
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)
            
            elif cp_rg == 2:
                if var[0] == var1 and var[1] == var2:
                    p = patches.Rectangle((bmin[0], bmin[1]),
                                          (bmax[0] - bmin[0]) + 0.99,
                                          (bmax[1] - bmin[1]) + 0.99,
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)
                
                elif var[1] == var1 and var[0] == var2:
                    p = patches.Rectangle((bmin[1], bmin[0]),
                                          (bmax[1] - bmin[1]) + 0.99,
                                          (bmax[0] - bmin[0]) + 0.99,
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)
        
        if cp is None:
            plt.gca().set_title('rules activations')
        else:
            plt.gca().set_title('rules cp%s activations' % str(cp))
        
        plt.gca().axis([-0.1, nb_bucket + 0.1, -0.1, nb_bucket + 0.1])
    
    def plot_pred(self, x, y, var1, var2, cmap=None,
                  vmin=None, vmax=None, add_points=True,
                  add_score=False):
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
               Number of the column of the first variable

        var2 : {int type}
               Number of the column of the second variable

        cmap : {colormap object}, optional
               Colormap used for the graphic

        vmax, vmin : {float type}, optional
                     Parameter of the range of the colorbar

        add_points: {boolean type}, optional
                    Option to add the discrete scatter of y

        add_score : {boolean type}, optional
                    Option to add the score on the graphic

        -------
        Draw the graphic
        """
        nb_bucket = self.get_param('nb_bucket')
        x_discretized = self.discretize(x)
        selected_rs = self.get_param('selected_rs')
        yapp = self.get_param('y')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        
        x1 = x_discretized[:, var1]
        x2 = x_discretized[:, var2]
        
        xx, yy = np.meshgrid(range(nb_bucket),
                             range(nb_bucket))
        
        if cmap is None:
            cmap = plt.cm.get_cmap('coolwarm')
        
        z = selected_rs.predict(yapp, np.c_[np.round(xx.ravel()), np.round(yy.ravel())],
                                ymean, ystd)
        
        if vmin is None:
            vmin = min(z)
        if vmax is None:
            vmax = max(z)
        
        z = z.reshape(xx.shape)
        
        plt.contourf(xx, yy, z, cmap=cmap, alpha=.8, vmax=vmax, vmin=vmin)
        
        if add_points:
            area = map(lambda b:
                       map(lambda a:
                           np.extract(np.logical_and(x1 == a, x2 == b),
                                      y).mean(), range(nb_bucket)),
                       range(nb_bucket))
            
            area_len = map(lambda b:
                           map(lambda a:
                               len(np.extract(np.logical_and(x1 == a, x2 == b),
                                              y)) * 10, range(nb_bucket)),
                           range(nb_bucket))
            
            plt.scatter(xx, yy, c=area, s=area_len, alpha=1.0,
                        cmap=cmap, vmax=vmax, vmin=vmin)
        
        plt.title('RIPE prediction')
        
        if add_score:
            score = self.score(x, y)
            plt.text(nb_bucket - .70, .08, ('%.2f' % str(score)).lstrip('0'),
                     size=20, horizontalalignment='right')
        
        plt.axis([-0.01, nb_bucket - 0.99, -0.01, nb_bucket - 0.99])
        plt.colorbar()
    
    def plot_counter_variables(self):
        """
        Function plots a graphical counter of variables used in rules.
        """
        ruleset = self.get_param('selected_rs')
        counter = get_variables_count(ruleset)
        
        x_labels = map(lambda item: item[0], counter)
        values = map(lambda item: item[1], counter)
        
        f = plt.figure()
        ax = plt.subplot()
        
        g = sns.barplot(y=x_labels, x=values, ax=ax, ci=None)
        g.set(xlim=(0, max(values) + 1), ylabel='Variable', xlabel='Count')
        
        return f
    
    def plot_counter(self):
        """
        Function plots a graphical counter of variables used in rules by modality.
        """
        nb_bucket = self.get_param('nb_bucket')
        y_labels, counter = self.make_count_matrix(return_vars=True)
        
        x_labels = map(lambda i: str(i), range(nb_bucket))
        
        f = plt.figure()
        ax = plt.subplot()
        
        g = sns.heatmap(counter, xticklabels=x_labels, yticklabels=y_labels,
                        cmap='Reds', linewidths=.05, ax=ax, center=0.0)
        g.xaxis.tick_top()
        plt.yticks(rotation=0)
        
        return f
    
    def plot_dist(self, metric=dist):
        """
        Function plots a graphical correlation of rules.
        """
        ruleset = self.get_param('selected_rs')
        rules_names = ruleset.get_rules_name()
        
        activation_list = map(lambda rules: rules.get_pred_vect(), ruleset)
        pred_mat = np.matrix(activation_list)
        
        dist_vect = scipy_dist.pdist(pred_mat, metric=metric)
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
        sns.heatmap(dist_mat, cmap=cmap, ax=ax,
                    vmax=vmax, vmin=vmin, center=1.,
                    square=True, xticklabels=rules_names,
                    yticklabels=rules_names, mask=mask)
        
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        
        return f
    
    def plot_intensity(self):
        """
        Function plots a graphical counter of variables used in rules.
        """
        y_labels, counter = self.make_count_matrix(return_vars=True)
        intensity = self.make_count_matrix(add_pred=True)
        
        nb_bucket = self.get_param('nb_bucket')
        x_labels = map(lambda i: str(i), range(nb_bucket))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.divide(intensity, counter)
        
        val[np.isneginf(val)] = np.nan
        val = np.nan_to_num(val)
        
        f = plt.figure()
        ax = plt.subplot()
        
        g = sns.heatmap(val, xticklabels=x_labels, yticklabels=y_labels,
                        cmap='bwr', linewidths=.05, ax=ax, center=0.0)
        g.xaxis.tick_top()
        plt.yticks(rotation=0)
        
        return f
    
    def make_count_matrix(self, add_pred=False, return_vars=False):
        """
        Return a count matrix of each variable in each modality
        """
        ruleset = self.get_param('selected_rs')
        nb_bucket = self.get_param('nb_bucket')
        
        counter = get_variables_count(ruleset)
        
        vars_list = map(lambda item: item[0], counter)
        
        count_mat = np.zeros((nb_bucket, len(vars_list)))
        str_id = []
        
        for rg in ruleset:
            cd = rg.conditions
            var_name = cd.get_param('features_name')
            bmin = cd.get_param('bmin')
            bmax = cd.get_param('bmax')
            
            for j in range(len(var_name)):
                if type(bmin[j]) != str:
                    for b in range(int(bmin[j]), int(bmax[j]) + 1):
                        var_id = vars_list.index(var_name[j])
                        if add_pred:
                            count_mat[b, var_id] += rg.get_param('pred')
                        else:
                            count_mat[b, var_id] += 1
                else:
                    str_id += [vars_list.index(var_name[j])]
        
        vars_list = [i for j, i in enumerate(vars_list) if j not in str_id]
        count_mat = np.delete(count_mat.T, str_id, 0)
        
        if return_vars:
            return vars_list, count_mat
        else:
            return count_mat
    
    def make_selected_df(self):
        """
        Returns
        -------
        selected_df : {DataFrame type}
                      DataFrame of selected RuleSet for presentation
        """
        selected_rs = self.get_param('selected_rs')
        df = selected_rs.to_df()
        
        df.rename(columns={"Cov": "Coverage", "Pred": "Prediction",
                           'Var': 'Variance', 'Crit': 'Criterion'},
                  inplace=True)
        
        df['Conditions'] = map(lambda rule: make_condition(rule), selected_rs)
        selected_df = df[['Conditions', 'Coverage',
                          'Prediction', 'Variance',
                          'Criterion']].copy()
        
        no_rules_df = self.get_complementary_rule()
        if no_rules_df is not None:
            selected_df = selected_df.append(no_rules_df)
        
        selected_df['Coverage'] = selected_df.Coverage.round(2)
        selected_df['Prediction'] = selected_df.Prediction.round(2)
        selected_df['Variance'] = selected_df.Variance.round(2)
        selected_df['Criterion'] = selected_df.Criterion.round(2)
        
        return selected_df
    
    def get_complementary_rule(self):
        """
        Returns
        -------
        norule_df : {DataFrame type or None}
                    DataFrame with the no activated rule.
        """
        y = self.get_param('y')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        yreal = self.get_param('yreal')
        method = self.get_param('calcmethod')
        rs = self.get_param('selected_rs')
        
        sum_vect = np.sum(map(lambda rg: rg.get_activation(), rs), axis=0)
        sum_vect = np.array(map(lambda x: bool(x), sum_vect))
        active_vect = np.logical_not(sum_vect).astype(int)
        
        cov = calc_coverage(active_vect)
        
        if cov > 0.0:
            pred = calc_prediction(active_vect, y)
            var = calc_variance(active_vect, y)
            
            pred_vect = active_vect * pred
            cplt_val = calc_prediction(1 - active_vect, y)
            np.place(pred_vect, pred_vect == 0, cplt_val)
            crit = calc_crit(pred_vect, yreal, ymean, ystd, method)
            
            norule_df = pd.DataFrame(index=['R ' + str(len(rs))],
                                     columns=['Conditions', 'Coverage',
                                              'Prediction', 'Variance',
                                              'Criterion'])
            
            norule_df['Conditions'] = 'No rule activated'
            norule_df['Coverage'] = cov
            norule_df['Prediction'] = pred
            norule_df['Variance'] = var
            norule_df['Criterion'] = crit
            
            return norule_df
        
        else:
            return None
    
    """------   Getters   -----"""
    
    def get_param(self, param):
        """
        To get the parameter param
        """
        assert type(param) == str, 'Must be a string'
        if hasattr(self, param):
            return getattr(self, param)
        else:
            return None
    
    """------   Setters   -----"""
    
    def set_params(self, **parameters):
        """
        To set a new parameter
        Example:
        --------
        o.set_params(new_param=val_new_param)
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
