# -*- coding: utf-8 -*-
"""
Created on 22 sept. 2016
@author: VMargot
"""
import sys
import math
import copy
import operator
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


def make_rules(feature_name, feature_index, X, y, method, sini_crit,
               th, cov_min, cov_max, yreal, ymean, ystd):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.

    Parameters
    ----------
    feature_name : {str type}
                   Name of the feature

    feature_index : {int type}
                    Columns index of the feature

    X : {array-like or discretized matrix, shape = [n, d]}
        The training input samples after discretization.

    y : {array-like, shape = [n]}
        The normalized target values (real numbers).

    method : {string type}
             The methode mse_function or mse_function criterion

    sini_crit : {string type}
                The significance test

    th : {float type such as 0 < th < 1}
         The threshold for the type 1 error

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
               the list of all suitable rules on the choosen feature.
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
                conditions = RuleCondditions(features_name=[feature_name],
                                             features_index=[feature_index],
                                             bmin=[bmin],
                                             bmax=[bmax],
                                             xmax=[max(values)],
                                             xmin=[min(values)],
                                             values=values)

                rule = Rule(conditions)
                rules_list.append(eval_rule(rule, X, y, method, sini_crit, th,
                                            cov_min, cov_max, yreal, ymean, ystd))

        else:
            bmax = bmin
            conditions = RuleCondditions(features_name=[feature_name],
                                         features_index=[feature_index],
                                         bmin=[bmin],
                                         bmax=[bmax],
                                         xmax=[max(values)],
                                         xmin=[min(values)],
                                         values=values)

            rule = Rule(conditions)
            rules_list.append(eval_rule(rule, X, y, method, sini_crit, th,
                                        cov_min, cov_max, yreal, ymean, ystd))

    rules_list = filter(None, rules_list)
    return rules_list


def eval_rule(rule, X, y, method, sini_crit, th,
              cov_min, cov_max, yreal, ymean, ystd):
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

    sini_crit : {string type}
                The significance test

    th : {float type such as 0 < th < 1}
         The threshold for the type 1 error

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
    None : if the rule does not verified criteria

    rule : {rule type}
             rule with all statistics calculated

    """
    rule.calc_stats(x=X, y=y, method=method,
                    sini_crit=sini_crit, th=th,
                    cov_min=cov_min, cov_max=cov_max,
                    yreal=yreal, ymean=ymean, ystd=ystd)

    if rule.get_param('out') is False:
        return rule
    else:
        return None


def find_upcp(rule, rulesset_cp1, cp):
    """
    Calculation of all statistics of an rules

    Parameters
    ----------
    rule : {rule type}
             An rule object

    rulesset_cp1 : {rulesSet type}
                 A set of rule of complecity 1

    cp : {int type, cp > 1}
         A given complexity

    Return
    ------
    rules_list : {list type}
                 List of rule made by intersection of rule with
                 rules from the rules set rulesset_cp1.

    """
    if cp == 2:
        i = rulesset_cp1.rules.index(rule)
        rules_list = map(lambda rules_cp1: rule.intersect(rules_cp1, cp),
                         rulesset_cp1[i+1:])
        return rules_list
                
    else:
        rules_list = map(lambda rules_cp1: rule.intersect(rules_cp1, cp),
                         rulesset_cp1)
        return rules_list


def union_test(rulesset, rule, j, inter_max):
    """
    Test to add a new rule (rule) to a set of rule
    (rulesset)

    Parameters
    ----------
    rulesset : {rulesSet type}
             An rule object

    rule : {rule type}
             A set of rule of complecity 1

    j : {int type or None}
        If j is not not we drop the j-th rule of rulesset
        to try to add the new rule

    inter_max : {float type, 0 <= inter_max <= 1}
                Maximal rate of intersection

    Return
    ------
    rulesset_copy : {rulesSet type}
                  A set of rules with a new rule if the
                  the intersection test is satisfied

    None : If the intersection test between the new rule
           and the set of exepert is not satisfied

    """
    rulesset_copy = copy.deepcopy(rulesset)
    if j is not None:
        rulesset_copy.pop(j)
        if len(rulesset_copy) > 1:
            for i in range(len(rulesset_copy)):
                rules = rulesset_copy[i]
                utest = rule.union_test(rules.get_activation(), inter_max)
                if utest is False:
                    return None

        if rule.union_test(rulesset_copy.calc_activation(), inter_max):
            rulesset_copy.insert(j, rule)
            return rulesset_copy
        else:
            return None
    else:
        if rule.union_test(rulesset_copy.calc_activation(), inter_max):
            rulesset_copy.append(rule)
            return rulesset_copy
        else:
            return None


def calc_rulesset_crit(rulesset, yapp, yreal, ymean, ystd, method):
    """
    Calculation of the criterium of a set of rule

    Parameters
    ----------
    rulesset : {rulesSet type}
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
             The methode mse_function or mse_function criterion

    Return
    ------
    crit : {float type}
           The value of the criterium for the method
    """
    pred_vect = rulesset.calc_pred(y_app=yapp)
    crit = calc_crit(pred_vect, yreal, ymean, ystd, method)
    return crit


def get_variables_count(rulesset):
    """
    Get a counter of all different features in the rulesset

    Parameters
    ----------
    rulesset : {rulesSet type}
             A set of rules

    Return
    ------
    count : {Counter type}
            Counter of all diffrent features in the rulesset
    """
    col_varulesset = map(lambda rg: rg.conditions.get_param('features_name'),
                         rulesset)
    varulesset_list = reduce(operator.add, col_varulesset)
    count = Counter(varulesset_list)

    count = count.most_common()
    return count


def dist(u, v):
    """
    Compute the distance bewteen two prediction vector

    Parameters
    ----------
    u,v : {array type}
          A precictor vector. It means a sparse array with two
          different values 0, if the rule is not active
          and the prediction is the rule is active.

    Return
    ------
    Distance between u and v
    """
    assert len(u) == len(v),\
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
                A precictor vector. It means a sparse array with two
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
                A precictor vector. It means a sparse array with two
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
                A precictor vector. It means a sparse array with two
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


def calc_crit(pred_vect, y,
              ymean=0, ystd=1,
              method='mse_function'):
    """
    Compute the criterium

    Parameters
    ----------
    pred_vect : {array type}
                A precictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    ymean : {float type}
            The mean of y.

    ystd : {float type}
           The standard deviation of y.

    method : {string type}
             The methode mse_function or mse_function criterion

    Return
    ------
    crit : {float type}
           Criterium value
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
    The bound for the conditional rulesectation to be significant
    """
    nb_activation = np.sum(np.extract(active_vect, np.isfinite(y)))
    num = np.sqrt(nb_activation)
    deno = np.nanstd(y)
    ratio = deno / num
    eps = 1 - th/2.0
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
    The bound for the conditional rulesectation to be significant
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
    The bound for the conditional rulesectation to be significant
    """
    sub_y = np.extract(active_vect, y)
    y_max = np.nanmax(sub_y)
    y_min = np.nanmin(sub_y)
    n = np.sum(active_vect)

    num = (y_max - y_min) * np.sqrt(np.log(2./th))
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
    The bound for the conditional rulesectation to be significant
    """
    sub_y = np.extract(active_vect, y)
    y_max = np.nanmax(sub_y)
    v = np.nansum(sub_y ** 2)
    n = np.sum(active_vect)

    val1 = y_max * np.log(2./th)
    val2 = 72.0 * v * np.log(2./th)
    return 1./(6. * n) * (val1 + np.sqrt(val1 ** 2 + val2))


def calc_coverage(vect):
    """
    Compute the coverage rate of an activation vecto

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
    Compute the empirical conditional rulesectation of y
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
           the empirical conditional rulesectation of y
           knowing X
    """
    y_cond = np.extract(active_vect != 0, y)
    if sum(~np.isnan(y_cond)) == 0:
        return 0
    else:
        pred = np.nanmean(y_cond)
        return pred


def find_bins(xcol, nb_bucket):
    """
    Function used to find the bins to discretize xcol in nb_bucket modalities

    Parameters
    ----------
    xcol : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                number of modalities

    Return
    ------
    bins : {ndarray type}
           the bins for disretization (result from numpy percentile function)
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
           Serie to discretize

    nb_bucket : {int type}
                number of modalities

    bins : {ndarray type}, optional, default None
           if you have already calculate the bins for xcol

    Return
    ------
    xcol_discretized : {Series type}
                       the discretization of xcol
    """
    if xcol.dtype.type != np.object_:
        # extraction of the list of xcol values
        notnan_vect = np.extract(np.isfinite(xcol), xcol)
        nan_index = ~np.isfinite(xcol)
        # Test if xcol have more than nb_bucket different values
        if len(set(notnan_vect)) >= nb_bucket:
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


class RuleCondditions(object):
    """
    Class for binary rule condition
    """
    def __init__(self, features_name, features_index,
                 bmin, bmax, xmin, xmax, values=list([])):
        
        self.features_name = features_name
        self.features_index = features_index
        if type(bmin[0]) != np.string_:
            assert all(map(lambda a, b: a <= b, bmin, bmax)), \
                'Bmin must be smaller or equal than bmax (%s)' % features_name
        self.bmin = bmin
        self.bmax = bmax
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
        to_hash = [(self.features_index[i], self.bmin[i], self.bmax[i])
                   for i in range(len(self.features_index))]
        to_hash = tuple(to_hash)
        return hash(to_hash)

    def transform(self, xmat):
        """
        Transform a matrix xmat into an activation vector.
        It means an array of 0 and 1. 0 if the condition is not
        satisfied and 1 otherwise.

        Parameters
        ----------
        xmat: {array-like matrix, shape=(n_samples, n_features)}
              Imput data
        
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
        assert type(param) == str,\
            'Must be a string'
        
        return getattr(self, param)
    
    def get_attr(self):
        """
        To get a list of attributes of self.
        It is usefull to quickly create a RuleCondditions
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
        
        assert rule_conditions.__class__ == RuleCondditions, \
            'Must be a RuleConddition object'
        
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
            return 1*intersection
    
    def test_variables(self, rule):
        """
        Test to know if a rule (self) and an other (rule)
        have conditions on the same features.
        """
        c1 = self.conditions
        c2 = rule.conditions
        
        c1_name = c1.get_param('features_name')
        c2_name = c2.get_param('features_name')

        if self.get_param('cp') == 1 and rule.get_param('cp') == 1:
            return c1_name[0] == c2_name[0]

        elif self.get_param('cp') > 1 and rule.get_param('cp') == 1:
            return any(map(lambda var: c2_name[0] == var,
                           c1_name))

        elif self.get_param('cp') == 1 and rule.get_param('cp') > 1:
            return any(map(lambda var: c1_name[0] == var,
                           c2_name))

        else:
            if len(set(c1_name).intersection(c2_name)) != 0:
                return False
            else:
                return True
    
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
        Compute an RuleConddition object from the intersection of an rule
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
                
                new_conditions = RuleCondditions(features_name=conditions_list[0],
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
                   sini_crit='hoeffding', th=0.05,
                   cov_min=0.01, cov_max=0.5,
                   yreal=None, ymean=0, ystd=1):
        """
        Calculation of all statistics of an rules

        Parameters
        ----------
        x : {array-like or discretized matrix, shape = [n, d]}
            The training input samples after discretization.

        y : {array-like, shape = [n]}
            The normalized target values (real numbers).

        method : {string type}
                 The methode mse_function or mse_function criterion

        sini_crit : {string type}
                    The significance test

        th : {float type such as 0 < th < 1}, default 0.05
             The threshold for the type 1 error

        cov_min : {float type such as 0 <= covmin <= 1}, default 0.01
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
        None : if the rule does not verified criteria
        """
        self.set_params(out=False)
        active_vect = self.calc_activation(x=x)

        self.set_params(activation=active_vect)

        cov = calc_coverage(active_vect)
        self.set_params(cov=cov)
        
        if cov < cov_min or cov > cov_max:
            self.set_params(out=True)
            self.set_params(reason='Cov')
            return

        else:
            pred = calc_prediction(active_vect, y)
            self.set_params(pred=pred)

            if yreal is None:
                yreal = y

            pred_vect = active_vect * pred
            cplt_val = calc_prediction(1-active_vect, y)
            np.place(pred_vect, pred_vect == 0, cplt_val)
            
            rez = calc_crit(pred_vect, yreal, ymean, ystd, method)
            self.set_params(crit=rez)

            if sini_crit == 'zscore':
                sini_th = calc_zscore(active_vect, y, th)
            elif sini_crit == 'tscore':
                sini_th = calc_tscore(active_vect, y, th)
            elif sini_crit == 'hoeffding':
                sini_th = calc_hoeffding(active_vect, y, th)
            elif sini_crit == 'bernstein':
                sini_th = calc_bernstein(active_vect, y, th)
            else:
                sini_th = 0

            self.set_params(th=sini_th)
            if abs(pred) < sini_th:
                self.set_params(out=True)
                self.set_params(reason='Concentration')
                return

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

        nan_val = np.argwhere(np.isnan(y))
        if len(nan_val) > 0:
            new_index = y.dropna(inplace=True).index
            pred_vect = pred_vect.loc[new_index]
        if score_type == 'Classification':
            th_val = (min(y) + max(y)) / 2.0
            pred_vect = np.array(map(lambda p: min(y) if p < th_val else max(y), pred_vect))
            return accuracy_score(y, pred_vect)
        
        elif score_type == 'Regression':
            return r2_score(y, pred_vect, sample_weight=sample_weight,
                            multioutput='variance_weighted')            

    def make_name(self, num, learning=None):
        """
        Add an attribut name to self

        Parameters
        ----------
        num : {int type}
              index of the rule in an rulesSet

        learning : {leanring type}, default None
                   If leaning is not None the name of self will
                   be definied with the name of learning
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


class RulesSet(object):
    """
    Class for a rulesSet. It's a kind of list of rule object
    """
    def __init__(self, param):
        if type(param) == list:
            self.rules = param
        elif type(param) == RulesSet:
            self.rules = param.get_rules()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'rulesSet: %s rules' % str(len(self.rules))

    def __gt__(self, val):
        return map(lambda rg: rg > val, self.rules)
    
    def __lt__(self, val):
        return map(lambda rg: rg < val, self.rules)
    
    def __ge__(self, val):
        return map(lambda rg: rg >= val, self.rules)
    
    def __le__(self, val):
        return map(lambda rg: rg <= val, self.rules)
    
    def __add__(self, rulesset):
        return self.extend(rulesset)
    
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
        :param rule:
        :return:
        """
        assert rule.__class__ == Rule, 'Must be a rule object (try extend)'
        self.rules.append(rule)
    
    def extend(self, rulesset):
        """
        :param rulesset:
        :return:
        """
        assert rulesset.__class__ == RulesSet, 'Must be a rulesSet object'
        'rulesSet must have the same Learning object'
        rules_list = rulesset.get_rules()
        self.rules.extend(rules_list)
        return self
    
    def insert(self, idx, rule):
        """
        :param idx:
        :param rule:
        :return:
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.rules.insert(idx, rule)
    
    def pop(self, idx=None):
        """
        :param idx:
        :return:
        """
        self.rules.pop(idx)
    
    def extract_greater(self, param, val):
        """
        :param param:
        :param val:
        :return:
        """
        rules_list = filter(lambda rg: rg.get_param(param) > val, self)        
        return RulesSet(rules_list)
    
    def extract_least(self, param, val):
        """
        :param param:
        :param val:
        :return:
        """
        rules_list = filter(lambda rg: rg.get_param(param) < val, self)
        return RulesSet(rules_list)
    
    def extract_cp(self, cp):
        """
        :param cp:
        :return:
        """
        rules_list = filter(lambda rg: rg.get_param('cp') == cp, self)
        return RulesSet(rules_list)
    
    def extract(self, param, val):
        """
        :param param:
        :param val:
        :return:
        """
        rules_list = filter(lambda rg: rg.get_param(param) == val, self)
        return RulesSet(rules_list)
    
    def index(self, rule):
        """
        :param rule:
        :return:
        """
        assert rule.__class__ == Rule, 'Must be a rule object'
        self.get_rules().index(rule)

    def replace(self, idx, rule):
        """
        :param idx:
        :param rule:
        :return:
        """
        self.rules.pop(idx)
        self.rules.insert(idx, rule)
    
    def sort_by(self, crit, maximized):
        """
        :param crit:
        :param maximized:
        :return:
        """
        self.rules.sort(key=lambda x: x.get_param(crit),
                        reverse=maximized)

    def to_df(self, cols=None):
        """
        To transform an rulesSet into a pandas DataFrame
        """
        if cols is None:
            cols = ['Features_Name', 'BMin', 'BMax',
                    'Cov', 'Pred', 'Crit', 'Th']

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
        using an rule based partion
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

        # Calculation of the conditional rulesectation in each cell
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
        Add an attribut name at each rule of self
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
                 2 diffretns values
                 Choose among the mse_function and mse_function criterion

        sinicrit : {string type} default bernstein if the number of row is
                   greatter than 30 else tscore
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

        nb_candidats : {int type} default 300
                    Choose the number of candidats to increase complexity

        intermax : {float type such as 0 <= intermax <= 1} default 1
                   Choose the maximal intersection rate begin a rule and
                   a current selected rulesSet
        
        nb_jobs : {int type} default number of core -2
                  Select the number of CPU used

        fullselection : {boolean type} default True
                        Choose if the selection is among all complexity (True)
                        or a selection by complexity (False)
        """
        self.selected_rs = RulesSet([])
        self.rulesset = RulesSet([])
        self.bins = dict()

        for arg, val in parameters.items():
            setattr(self, arg, val)

        if hasattr(self, 'th') is False:
            self.th = 0.05

        if hasattr(self, 'nb_jobs') is False:
            self.nb_jobs = -2

        if hasattr(self, 'nb_candidats') is False:
            self.nb_candidats = 300

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

        Returns
        -------
        return : {Learning type}
                 return self
        """

        # Check type for data
        X = check_array(X, dtype=None, force_all_finite=False)
        y = check_array(y, dtype=None, ensure_2d=False,
                        force_all_finite=False)

        # Creation of data-driven parameters
        if hasattr(self, 'nb_bucket') is False:
            nb_bucket = max(5, int(np.sqrt(pow(X.shape[0],
                                              1./X.shape[1]))))

            nb_bucket = min(nb_bucket, X.shape[0])
            self.set_params(nb_bucket=nb_bucket)

        if hasattr(self, 'covmin') is False:
            try:
                covmin = 1.0 / (pow(self.get_param('nb_bucket'),
                                    X.shape[1]))
                covmin = max(sys.float_info.epsilon, covmin)

            # For a very high dimension
            except OverflowError:
                covmin = sys.float_info.epsilon

            self.set_params(covmin=covmin)

        if hasattr(self, 'covmax') is False:
            covmax = 1.0 / np.log(self.get_param('nb_bucket'))
            covmax = min(0.99, covmax)

            self.set_params(covmax=covmax)

        if hasattr(self, 'sinicrit') is False:
            if y.shape[0] > 30:
                sinicrit = 'bernstein'  # 'zscore' 'hoeffding' 'bernstein'
            else:
                # We use a t-test for the small data
                sinicrit = 'tscore'

            self.set_params(sinicrit=sinicrit)

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
            features_name = map(lambda i: 'X'+str(i), features_index)

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

        # ------------
        # SEEKING PART
        # ------------

        self.calc_cp1()
        rulesset = self.get_param('rulesset')

        if len(rulesset) > 0:
            for cp in range(2, complexity + 1):
                if len(selected_rs.extract_cp(cp)) == 0:
                    # seeking a set of rules with a comlexity cp
                    rulesset_cpup = self.up_complexity(cp)

                    if len(rulesset_cpup) > 0:
                        rulesset += rulesset_cpup
                        self.set_params(rulesset=rulesset)
                    else:
                        print('No rules for complexity %s' % str(cp))
                        break

                    rulesset.sort_by('crit', maximized)
            self.set_params(rulesset=rulesset)

            # --------------
            # SELECTION PART
            # --------------
            print('----- Selection ------')
            selection_type = self.get_param('fullselection')
            if selection_type:
                selected_rs = self.select_rules(0)
            else:
                selected_rs = self.get_param('selected_rs')
                for cp in range(1, complexity+1):
                    selected_rs += self.select_rules(cp)

            rulesset.make_rule_names()
            self.set_params(rulesset=rulesset)
            selected_rs.make_rule_names()
            self.set_params(selected_rs=selected_rs)

        else:
            print('No rules found !')

    def calc_cp1(self):
        """
        Compute all rules of complexity one and keep the best
        """

        features_name = self.get_param('features_name')
        features_index = self.get_param('features_index')
        X = self.get_param('X')
        method = self.get_param('calcmethod')
        sini_crit = self.get_param('sinicrit')
        th = self.get_param('th')
        y = self.get_param('y')
        yreal = self.get_param('yreal')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        cov_min = self.get_param('covmin')
        cov_max = self.get_param('covmax')

        jobs = min(len(features_name), self.get_param('nb_jobs'))

        if jobs == 1:
            rulesset = map(lambda var, idx: make_rules(var, idx, X, y, method,
                                                       sini_crit, th,
                                                       cov_min, cov_max,
                                                       yreal, ymean, ystd),
                           features_name, features_index)
        else:
            rulesset = Parallel(n_jobs=jobs, backend="multiprocessing")(
                delayed(make_rules)(var, idx, X, y, method, sini_crit, th,
                                    cov_min, cov_max, yreal, ymean, ystd)
                for var, idx in zip(features_name, features_index))

        rulesset = reduce(operator.add, rulesset)

        rulesset = RulesSet(rulesset)
        rulesset.sort_by('crit', self.get_param('maximized'))
        
        self.set_params(rulesset=rulesset)
    
    def up_complexity(self, cp):
        """
        Returns a rulesSet of rules with complexity=cp

        :rtype: rulesSet
        """
        nb_jobs = self.get_param('nb_jobs')
        X = self.get_param('X')
        method = self.get_param('calcmethod')
        sini_crit = self.get_param('sinicrit')
        th = self.get_param('th')
        y = self.get_param('y')
        yreal = self.get_param('yreal')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        cov_min = self.get_param('covmin')
        cov_max = self.get_param('covmax')

        rules_list = self.find_candidates(cp)

        if len(rules_list) > 0:
            if nb_jobs == 1:
                rulesset = map(lambda rule: eval_rule(rule, X, y, method, sini_crit, th,
                                                      cov_min, cov_max, yreal, ymean, ystd),
                               rules_list)
            else:
                rulesset = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
                    delayed(eval_rule)(rule, X, y, method, sini_crit, th,
                                       cov_min, cov_max, yreal, ymean, ystd)
                    for rule in rules_list)

            rulesset = filter(None, rulesset)
            rulesset_cpup = RulesSet(rulesset)
            return rulesset_cpup
        else:
            return []

    def select_candidates(self, rules_cp, cp, cov_min=0.05):
        """
        Returns a selection of candidates to increase complexity
        for a given complexity (cp) and a min coverage (cov_min)
        """
        rulesset = self.get_param('rulesset')
        sub_rulesset = rulesset.extract_cp(rules_cp)

        bool_vect = map(lambda rg: rg.get_param('cov') >=
                        0.5 * pow(cov_min, float(rg.get_param('cp')) / cp),
                        sub_rulesset)

        extract = np.extract(bool_vect, sub_rulesset)
        rules_list = list(extract)

        nb_candidats = self.get_param('nb_candidats')
        if len(rules_list) > nb_candidats:
            rulesset_candidats = RulesSet(list(rules_list))

            pos_rulesset = rulesset_candidats.extract_greater('pred', 0)
            neg_rulesset = rulesset_candidats.extract_least('pred', 0)

            id_pos = float(len(pos_rulesset)) / len(rules_list) * nb_candidats
            id_neg = float(len(neg_rulesset)) / len(rules_list) * nb_candidats

            rules_list = pos_rulesset[:int(id_pos)]
            rules_list += neg_rulesset[:int(id_neg)]

        rulesset = RulesSet(list(rules_list))
        return rulesset
    
    def find_candidates(self, cp):
        """
        Returns the intersection of all suitable rules
        for a given complexity (cp) and a min coverage (cov_min)
        """
        cov_min = self.get_param('covmin')

        rulesset_cp1 = self.select_candidates(1, cp, cov_min)

        rulesset_candidate = self.select_candidates(cp - 1, cp, cov_min)

        if len(rulesset_candidate) > 0:
            rules_list = self.find_complexe_rules(cp, rulesset_cp1,
                                                  rulesset_candidate)

            return rules_list
        else:
            return []
    
    def find_complexe_rules(self, cp, rulesset_cp1, rulesset_candidate):
        """
        :param cp:
        :param rulesset_cp1:
        :param rulesset_candidate:
        :return:
        """
        nb_jobs = self.get_param('nb_jobs')

        if nb_jobs == 1:
            rules_list = map(lambda rule: find_upcp(rule, rulesset_cp1, cp),
                             rulesset_candidate)
        else:
            rules_list = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
                delayed(find_upcp)(rule, rulesset_cp1, cp)
                for rule in rulesset_candidate)

        rules_list = reduce(operator.add, rules_list)

        rules_list = filter(None, rules_list)  # to drop bad rules
        rules_list = list(set(rules_list))  # to drop duplicates
        return rules_list
    
    def select_rules(self, cp):
        """
        Returns a subset of a given rulesSet.
        This subset minimizes the empirical contrast on the learning set
        """
        maximized = self.get_param('maximized')
        rulesset = self.get_param('rulesset')

        if cp > 0:
            sub_rulesset = rulesset.extract_cp(cp)
        else:
            sub_rulesset = copy.deepcopy(rulesset)

        sub_rulesset.sort_by('crit', maximized)
        selected_rs = self.minimized_risk(sub_rulesset)

        return selected_rs

    def minimized_risk(self, rulesset):
        """
        Returns a subset of a given rulesSet. This subset is seeking by
        minimization/maximization of the criterion on the training set
        """
        yapp = self.get_param('y')
        yreal = self.get_param('yreal')
        ystd = self.get_param('ystd')
        ymean = self.get_param('ymean')
        method = self.get_param('calcmethod')
        maximized = self.get_param('maximized')
        inter_max = self.get_param('intermax')

        win = 50
        th = 0.75

        # Then optimization
        selected_rs = RulesSet(rulesset[:1])
        old_crit = calc_rulesset_crit(selected_rs, yapp, yreal,
                                      ymean, ystd, method)
        crit_evo = [old_crit]
        nb_rules = len(rulesset)

        k = 1

        for i in range(1, nb_rules):
            new_rules = rulesset[i]
            iter_list = [None]

            if len(selected_rs) > 1:
                iter_list += range(len(selected_rs))

            rulesset_list = []
            for j in iter_list:
                break_loop = False
                rulesset_copy = copy.deepcopy(selected_rs)
                if j is not None:
                    rulesset_copy.pop(j)
                    if new_rules.union_test(rulesset_copy.calc_activation(),
                                            inter_max):
                        if len(rulesset_copy) > 1:
                            for rules in rulesset_copy:
                                utest = new_rules.union_test(rules.get_activation(),
                                                             inter_max)
                                if not utest:
                                    break_loop = True
                                    break
                        if break_loop:
                            continue

                        rulesset_copy.insert(j, new_rules)
                        rulesset_list.append(rulesset_copy)
                    else:
                        continue

                else:
                    utest = map(lambda e: new_rules.union_test(e.get_activation(),
                                                               inter_max), rulesset_copy)
                    if all(utest) and new_rules.union_test(rulesset_copy.calc_activation(),
                                                           inter_max):
                        rulesset_copy.append(new_rules)
                        rulesset_list.append(rulesset_copy)

            if len(rulesset_list) > 0:
                crit_list = map(lambda e: calc_rulesset_crit(e, yapp, yreal,
                                                             ymean, ystd, method),
                                rulesset_list)

                if maximized:
                    rulesset_idx = int(np.argmax(crit_list))
                    if crit_list[rulesset_idx] >= old_crit:
                        selected_rs = copy.deepcopy(rulesset_list[rulesset_idx])
                        old_crit = crit_list[rulesset_idx]
                else:
                    rulesset_idx = int(np.argmin(crit_list))
                    if crit_list[rulesset_idx] <= old_crit:
                        selected_rs = copy.deepcopy(rulesset_list[rulesset_idx])
                        old_crit = crit_list[rulesset_idx]

            crit_evo.append(old_crit)

            # Stopping Criteria
            if len(crit_evo) > k * win:
                crit_delta = np.max(crit_evo)
                crit_diff = -np.diff(crit_evo[::win])
                crit_diff /= crit_delta
                k += 1
                if np.cumsum(crit_diff)[-1] > th:
                    break

        self.set_params(critlist=crit_evo)
        return selected_rs

    def predict(self, X, check_input=True):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        application of the selected rulesSet on X.
        
        Parameters
        ----------
        X : {array type or sparse matrix of shape = [n_samples, n_features]}
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparulessete matrix is provided, it will be
            converted into a sparulessete ``csr_matrix``.

        check_input : bool type

        Returns
        -------
        y : {array type of shape = [n_samples]}
            The predicted values.
        """
        y_app = self.get_param('y')

        X = self.validate_X_predict(X, check_input)
        x_copy = self.discretize(X)

        rulesset = self.get_param('selected_rs')
        ymean = self.get_param('ymean')
        ystd = self.get_param('ystd')
        
        pred_vect = rulesset.predict(y_app, x_copy, ymean, ystd)
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
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if hasattr(self, 'fitted') is False:
            raise AttributeError("Estimator not fitted, "
                                 "call 'fit' before rulesloiting the model.")

        if check_input:
            X = check_array(X, dtype=None, force_all_finite=False)

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
        the color is coresponding to the intensity of the prediction
        
        Parameters
        ----------
        var1 : {string type} 
               Name of the firulessett variable
        
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
            sub_rulesset = selected_rs.extract_cp(cp)
        else:
            sub_rulesset = selected_rs
        
        plt.plot()
        
        for rg in sub_rulesset:
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
                                          nb_bucket - 1,  # height
                                          hatch=hatch, facecolor=facecolor,
                                          alpha=alpha)
                    plt.gca().add_patch(p)
                    
                elif var[0] == var2:
                    p = patches.Rectangle((0, bmin[0]),
                                          nb_bucket - 1,
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

        plt.gca().axis([-0.1, nb_bucket - 0.9, -0.1, nb_bucket - 0.9])
            
    def plot_pred(self, x, y, var1, var2, cmap=None,
                  vmin=None, vmax=None, add_points=True,
                  add_score=False):
        """
        Plot the prediction zone of rules in a 2D plot
        
        Parameters
        ----------
        x : {array-like, sparulessete matrix}, shape=[n_samples, n_features]
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
                    Option to add the discret scatter of y
                    
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
            cmap = plt.cm.coolwarm

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
            plt.text(nb_bucket - .70, .08, ('%.2f' % score).lstrip('0'),
                     size=20, horizontalalignment='right')
        
        plt.axis([-0.01, nb_bucket - 0.99, -0.01, nb_bucket - 0.99])
        plt.colorbar()

    def plot_counter_variables(self):
        """
        :return:
        """
        rulesset = self.get_param('selected_rs')
        counter = get_variables_count(rulesset)

        x_labels = map(lambda item: item[0], counter)
        values = map(lambda item: item[1], counter)

        f = plt.figure()
        ax = plt.subplot()

        g = sns.barplot(y=x_labels, x=values, ax=ax, ci=None)
        g.set(xlim=(0, max(values) + 1), ylabel='Variable', xlabel='Count')

        return f

    def plot_counter(self):
        """
        Function plots a graphical counter of varaibles used in rules.
        """
        nb_bucket = self.get_param('nb_bucket')
        y_labels, counter = self.make_count_matrice(return_vars=True)

        x_labels = map(lambda i: str(i), range(nb_bucket))

        f = plt.figure()
        ax = plt.subplot()

        g = sns.heatmap(counter, xticklabels=x_labels, yticklabels=y_labels,
                        cmap='Reds', linewidths=.05, ax=ax)
        g.xaxis.tick_top()
        plt.yticks(rotation=0)

        return f

    def plot_dist(self, metric=dist):
        """
        Function plots a graphical correlation of rules.
        """

        if scipy_dist is not None:
            rulesset = self.get_param('selected_rs')
            rules_names = rulesset.get_rules_name()

            activation_list = map(lambda rules: rules.get_pred_vect(), rulesset)
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
        else:
            raise ImportError("The scipy package is required to run this function")

    def plot_intensity(self):
        """
        Function plots a graphical counter of varaibles used in rules.
        """

        y_labels, counter = self.make_count_matrice(return_vars=True)
        intensity = self.make_count_matrice(add_pred=True)

        nb_bucket = self.get_param('nb_bucket')
        x_labels = map(lambda i: str(i), range(nb_bucket))

        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.divide(intensity, counter)

        val[np.isneginf(val)] = np.nan
        val = np.nan_to_num(val)

        f = plt.figure()
        ax = plt.subplot()

        g = sns.heatmap(val, xticklabels=x_labels, yticklabels=y_labels,
                        cmap='bwr', linewidths=.05, ax=ax)
        g.xaxis.tick_top()
        plt.yticks(rotation=0)

        return f

    def make_count_matrice(self, add_pred=False, return_vars=False):
        """
        :param add_pred:
        :param return_vars:
        :return:
        """
        rulesset = self.get_param('selected_rs')
        nb_bucket = self.get_param('nb_bucket')

        counter = get_variables_count(rulesset)

        vars_list = map(lambda item: item[0], counter)

        count_mat = np.zeros((nb_bucket, len(vars_list)))

        for rg in rulesset:
            cd = rg.conditions
            var_name = cd.get_param('features_name')
            bmin = cd.get_param('bmin')
            bmax = cd.get_param('bmax')

            for j in range(len(var_name)):
                for b in range(int(bmin[j]), int(bmax[j])+1):
                    var_id = vars_list.index(var_name[j])
                    if add_pred:
                        count_mat[b, var_id] += rg.get_param('pred')
                    else:
                        count_mat[b, var_id] += 1

        if return_vars:
            return vars_list, count_mat.T
        else:
            return count_mat.T

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
