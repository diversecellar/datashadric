# -*- coding: utf-8 -*-
"""
Stochastics Functions Module
Comprehensive collection of statistical analysis utilities for hypothesis testing, distributions, and statistical inference
"""

# standard library imports
import calendar as cal

# third-party data science imports
import pandas as pd
import numpy as np

# statistical analysis imports
from statsmodels.stats.outliers_influence import variance_inflation_factor as smvif
import statsmodels.tools.tools as smtools
import statsmodels.stats.multicomp as smmulti
from statsmodels.tools.tools import add_constant as smac

# visualization imports
import matplotlib.pyplot as plt


def df_gaussian_checks(df_name, col_name, *args):
    """check if data follows gaussian distribution"""
    # usage: df_gaussian_checks(df, 'col_name')
    from scipy import stats
    data = df_name[col_name].dropna()
    
    # shapiro-wilk test
    stat, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk test for {col_name}:")
    print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    
    # q-q plot
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=ax)
    plt.title(f'Q-Q Plot for {col_name}')
    plt.show()
    
    return stat, p_value


def df_calc_conf_interval(moe_vals: dict, mean_val):
    """calculate confidence interval"""
    # usage: df_calc_conf_interval({'margin_of_error': 1.96}, mean_val=50)
    lower_bound = mean_val - moe_vals['margin_of_error']
    upper_bound = mean_val + moe_vals['margin_of_error']
    return {'lower': lower_bound, 'upper': upper_bound, 'mean': mean_val}


def df_calc_moe(stderr_val, z_score_cl):
    """calculate margin of error"""
    # usage: df_calc_moe(stderr_val=2.5, z_score_cl=1.96)
    margin_of_error = z_score_cl * stderr_val
    return {'margin_of_error': margin_of_error}


def df_calc_stderr(df_name, col_z, stddev_val):
    """calculate standard error"""
    # usage: df_calc_stderr(df, 'col_name', stddev_val=5.0)
    n = len(df_name[col_z].dropna())
    stderr = stddev_val / np.sqrt(n)
    return stderr


def df_calc_zscore(df_name, col_z, confidence_levels, mean_val, stddev_val):
    """calculate z-score for given confidence level"""
    # usage: df_calc_zscore(df, 'col_name', confidence_levels=95, mean_val=50, stddev_val=5.0)
    from scipy import stats
    alpha = 1 - confidence_levels / 100
    z_score = stats.norm.ppf(1 - alpha/2)
    return z_score