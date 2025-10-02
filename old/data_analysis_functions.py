# -*- coding: utf-8 -*-
"""
data analysis functions module
comprehensive collection of utility functions for data preprocessing, modeling, and evaluation
"""

# standard library imports
import re
import calendar as cal

# third-party data science imports
import pandas as pd
import numpy as np
import unidecode

# statistical analysis imports
from statsmodels.stats.outliers_influence import variance_inflation_factor as smvif
import statsmodels.formula.api as smfapi
import statsmodels.api as smapi
import statsmodels.tools.tools as smtools
import statsmodels.stats.multicomp as smmulti
from statsmodels.tools.tools import add_constant as smac
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# machine learning imports
import sklearn.linear_model as skllinmod
import sklearn.naive_bayes as sklnvbys
import sklearn.metrics as sklmtrcs
import sklearn.model_selection as sklmodslct

# visualization imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# dataframe utility functions
def df_print_row_and_columns(df_name):
    """print the number of rows and columns in a dataframe"""
    try:
        df_rows, df_columns = df_name.shape
    except Exception as e:
        df_rows, df_columns = df_name.to_frame().shape
    print("rows = {}".format(df_rows))
    print("columns = {}".format(df_columns))
    

def df_check_na_values(df_name, *args):
    """check for missing values in dataframe columns"""
    # usage: df_check_na_values(df) or df_check_na_values(df, ['col1', 'col2'])
    if not args:
        df_na = df_name.isna()
        mask = df_na == True
        masked = df_na[mask]
    else:
        df_na = df_name.isna()
        try:
            column_names = [arg for arg in args[0] if (isinstance(arg, str) and isinstance(args[0], list))]
        except Exception as e:
            print("need to be list of str type for args")
        for column in column_names:
            mask = df_na[column] == True
            masked = df_na[mask]
    print(masked)
    return df_name.isna()


def df_drop_na(df_name, ax: int):
    """drop missing values along specified axis"""
    # usage: df_drop_na(df, ax=0) # ax=0 for rows, ax=1 for columns
    if ax in [0, 1]:
        df_na_out = df_name.dropna(axis=ax)
        return df_na_out


def df_datetime_converter(df_name, col_datetime_lookup='date'):
    """convert columns containing date information to datetime format"""
    # usage: df_datetime_converter(df, 'date') or df_datetime_converter(df) 
    # defaults to all columns with 'date' in their name string
    for column in df_name.columns.tolist():
        if str(col_datetime_lookup) in str(column):
            print("yes")
            df_name[column] = pd.to_datetime(df_name[column])
    return df_name


def df_boxplotter(df_name, col_xplot, col_yplot, type_plot:int, *args):
    """create box plot to visualize outliers. type_plot: 0 for dist, 1 for money, 2 for general"""
    # usage: df_boxplotter(df, 'col_x', 'col_y', type_plot=0, 'horizontalalignment')
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6), dpi=100)
    sns.boxplot(x=df_name[col_xplot], y=df_name[col_yplot], ax=ax)
    matplotlib.pyplot.title('{} box plot to visualise outliers'.format(col_yplot))
    
    if type_plot == 0:
        matplotlib.pyplot.ylabel('{} in miles'.format(col_yplot))
    elif type_plot == 1:
        matplotlib.pyplot.ylabel('{} in $'.format(col_yplot))
    else:
        matplotlib.pyplot.ylabel('{}'.format(col_yplot))
    
    if args:
        matplotlib.pyplot.xticks(rotation=0, horizontalalignment=args[0])
    
    ax.yaxis.grid(True)
    matplotlib.pyplot.savefig("Boxplot_x-{}_y-{}.png".format(col_xplot, col_yplot))
    matplotlib.pyplot.show()


def df_explore_unique_categories(df_name, col):
    """print a dataframe with unique categories for each categorical variable"""
    # usage: df_explore_unique_categories(df, 'col_name')
    df_col_unique = df_name.drop_duplicates(subset=col, keep='first')
    return df_col_unique[col]


def df_histplotter(df_name, col_plot, type_plot:int, bins=10, *args):
    """create histogram plot. type_plot: 0 for dist, 1 for money"""
    # usage: df_histplotter(df, 'col_name', type_plot=0, bins=20)
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6), dpi=100)
    df_name[col_plot].hist(bins=bins, ax=ax)
    matplotlib.pyplot.title('{} histogram'.format(col_plot))
    
    if type_plot == 0:
        matplotlib.pyplot.xlabel('{} in miles'.format(col_plot))
    elif type_plot == 1:
        matplotlib.pyplot.xlabel('{} in $'.format(col_plot))
    else:
        matplotlib.pyplot.xlabel('{}'.format(col_plot))
    
    matplotlib.pyplot.ylabel('Frequency')
    ax.grid(True)
    matplotlib.pyplot.savefig("Histogram_{}.png".format(col_plot))
    matplotlib.pyplot.show()


def df_grouped_histplotter(df_name, col_groupby: str, col_plot: str, type_plot: int, bins=20):
    """create grouped histogram plots"""
    # usage: df_grouped_histplotter(df, 'col_groupby', 'col_plot', type_plot=0, bins=20)
    groups = df_name.groupby(col_groupby)
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6), dpi=100)
    
    for name, group in groups:
        group[col_plot].hist(bins=bins, alpha=0.7, label=name, ax=ax)
    
    matplotlib.pyplot.title('{} histogram grouped by {}'.format(col_plot, col_groupby))
    matplotlib.pyplot.xlabel(col_plot)
    matplotlib.pyplot.ylabel('Frequency')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


def df_mask_with_list(df, df_col, list_comp: list, mask_type: int):
    """mask dataframe with list comparison. mask_type: 0 for isin, 1 for not isin"""
    # usage: df_mask_with_list(df, 'col_name', ['val1', 'val2'], mask_type=0)
    if mask_type == 0:
        mask = df[df_col].isin(list_comp)
    else:
        mask = ~df[df_col].isin(list_comp)
    return df[mask]


def df_groupby_mask_operate(df, col_name_masker: str, col_name_operate: str, *args):
    """group by and perform operations on masked data"""
    # usage: df_groupby_mask_operate(df, 'col_groupby', 'col_operate', 'mean')
    grouped = df.groupby(col_name_masker)[col_name_operate]
    if args:
        return grouped.agg(args[0])
    return grouped.describe()


def df_cross_corr_check(df_name, cols_y: list, cols_x: list):
    """check cross-correlation between y and x variables"""
    # usage: df_cross_corr_check(df, ['col_y1', 'col_y2'], ['col_x1', 'col_x2'])
    correlation_matrix = df_name[cols_y + cols_x].corr()
    return correlation_matrix.loc[cols_y, cols_x]


def df_class_balance(df_filtered):
    """check class balance in filtered dataframe"""
    # usage: df_class_balance(df_filtered)
    value_counts = df_filtered.value_counts()
    percentages = df_filtered.value_counts(normalize=True) * 100
    balance_df = pd.DataFrame({
        'Count': value_counts,
        'Percentage': percentages
    })
    return balance_df


def df_grouped_barplotter(df_name, col_groupby: str, col_plot: str, type_plot: int):
    """create grouped bar plots"""
    # usage: df_grouped_barplotter(df, 'col_groupby', 'col_plot', type_plot=0)
    grouped_data = df_name.groupby(col_groupby)[col_plot].mean()
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6), dpi=100)
    grouped_data.plot(kind='bar', ax=ax)
    matplotlib.pyplot.title('{} by {}'.format(col_plot, col_groupby))
    matplotlib.pyplot.xlabel(col_groupby)
    matplotlib.pyplot.ylabel(col_plot)
    matplotlib.pyplot.xticks(rotation=45)
    matplotlib.pyplot.show()


def df_drop_dupes(df, col_dupes: int, *args):
    """drop duplicate rows based on specified columns"""
    # usage: df_drop_dupes(df) or df_drop_dupes(df, ['col1', 'col2'])
    if args:
        subset_cols = args[0] if isinstance(args[0], list) else [args[0]]
        return df.drop_duplicates(subset=subset_cols, keep='first')
    return df.drop_duplicates(keep='first')


def df_drop_col(df, col_name: str):
    """drop specified column from dataframe"""
    # usage: df_drop_col(df, 'col_name')
    if col_name in df.columns:
        return df.drop(columns=[col_name])
    return df


def df_scatterplotter(df_grouped, col_xplot, col_yplot):
    """create scatter plot between two variables"""
    # usage: df_scatterplotter(df, 'col_x', 'col_y')
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6), dpi=100)
    df_grouped.plot.scatter(x=col_xplot, y=col_yplot, ax=ax)
    matplotlib.pyplot.title('Scatter plot: {} vs {}'.format(col_xplot, col_yplot))
    matplotlib.pyplot.show()


def df_corr_check(df_name, col_y, col_x):
    """check correlation between two variables"""
    # usage: df_corr_check(df, 'col_y', 'col_x')
    correlation = df_name[[col_y, col_x]].corr().iloc[0, 1]
    return correlation


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
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=ax)
    matplotlib.pyplot.title(f'Q-Q Plot for {col_name}')
    matplotlib.pyplot.show()
    
    return stat, p_value


def df_calc_conf_interval(moe_vals:dict, mean_val):
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


def df_head(df_name, head_num: int):
    """display first n rows of dataframe"""
    # usage: df_head(df, head_num=5)
    return df_name.head(head_num)


def df_pairplot(data):
    """create pairplot for data exploration"""
    # usage: df_pairplot(df)
    sns.pairplot(data)
    matplotlib.pyplot.show()


def lr_check_homoscedasticity(fitted, resid, *args):
    """check homoscedasticity assumption in linear regression"""
    # usage: lr_check_homoscedasticity(fitted, resid)
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6))
    ax.scatter(fitted, resid)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values')
    matplotlib.pyplot.show()


def lr_check_normality(resid):
    """check normality of residuals"""
    # usage: lr_check_normality(resid)
    from scipy import stats
    stat, p_value = stats.shapiro(resid)
    print(f"Shapiro-Wilk test for residuals:")
    print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    return stat, p_value


def lr_qqplots_normality(resid):
    """create q-q plots to check normality of residuals"""
    # usage: lr_qqplots_normality(resid)
    from scipy import stats
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6))
    stats.probplot(resid, dist="norm", plot=ax)
    matplotlib.pyplot.title('Q-Q Plot of Residuals')
    matplotlib.pyplot.show()


def remove_whitespace(str_target: str):
    """remove whitespace from string"""
    # usage: remove_whitespace(' some text ')
    return str_target.replace(' ', '')


def remove_unicode(str_target: str):
    """remove unicode characters from string"""
    # usage: remove_unicode('café')
    try:
        clean_string = unidecode.unidecode(str_target)
        return clean_string
    except Exception as e:
        print(f"Error cleaning unicode: {e}")
        return str_target


def lr_post_hoc_test(df_name, col_response, col_predictor, alpha:float):
    """perform post-hoc test for anova"""
    # usage: lr_post_hoc_test(df, 'response_col', 'predictor_col', alpha=0.05)
    tukey_results = pairwise_tukeyhsd(
        endog=df_name[col_response],
        groups=df_name[col_predictor],
        alpha=alpha
    )
    print(tukey_results)
    return tukey_results


def lr_ols_model(df_name, col_response:str, col_cont_predictors:list, col_cat_predictors:list):
    """build ols regression model"""
    # usage: lr_ols_model(df, 'response_col', ['cont1', 'cont2'], ['cat1', 'cat2'])
    # add constant to categorical predictors
    
    # prepare formula
    predictors = col_cont_predictors + col_cat_predictors
    formula = f"{col_response} ~ {' + '.join(predictors)}"
    
    # fit model
    model = smfapi.ols(formula, data=df_name).fit()
    print(model.summary())
    
    return {
        'model': model,
        'formula': formula,
        'fitted_values': model.fittedvalues,
        'residuals': model.resid,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj
    }


def logr_predictor(df_name, log_regression_model: dict):
    """make predictions using logistic regression model"""
    # usage: logr_predictor(df, log_regression_model)
    model = log_regression_model['model']
    X = log_regression_model['X_test']
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities
    }


def logr_classifier(df_name, log_regression_model: dict):
    """classify using logistic regression model"""
    # usage: logr_classifier(df, log_regression_model)
    predictions = logr_predictor(df_name, log_regression_model)
    y_true = log_regression_model['y_test']
    y_pred = predictions['predictions']
    
    accuracy = sklmtrcs.accuracy_score(y_true, y_pred)
    precision = sklmtrcs.precision_score(y_true, y_pred, average='weighted')
    recall = sklmtrcs.recall_score(y_true, y_pred, average='weighted')
    f1 = sklmtrcs.f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def logr_train_test_split(df_name, col_response, col_predictor, test_size:float):
    """split data for logistic regression training and testing"""
    # usage: logr_train_test_split(df, 'response_col', 'predictor_col', test_size=0.2)
    X = df_name[col_predictor]
    y = df_name[col_response]
    
    X_train, X_test, y_train, y_test = sklmodslct.train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def ml_train_test_split(df_name, col_target, test_size:float):
    """generic train test split for machine learning"""
    # usage: ml_train_test_split(df, 'target_col', test_size=0.2)
    feature_cols = [col for col in df_name.columns if col != col_target]
    X = df_name[feature_cols]
    y = df_name[col_target]
    
    X_train, X_test, y_train, y_test = sklmodslct.train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def ml_naive_bayes_model(train_test_split_nm):
    """build naive bayes model"""
    # usage: ml_naive_bayes_model(train_test_split_nm)
    model = sklnvbys.GaussianNB()
    model.fit(train_test_split_nm['X_train'], train_test_split_nm['y_train'])
    
    train_predictions = model.predict(train_test_split_nm['X_train'])
    test_predictions = model.predict(train_test_split_nm['X_test'])
    
    return {
        'model': model,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'X_train': train_test_split_nm['X_train'],
        'X_test': train_test_split_nm['X_test'],
        'y_train': train_test_split_nm['y_train'],
        'y_test': train_test_split_nm['y_test']
    }


def ml_naive_bayes_metrics(naive_bayes_nm):
    """calculate metrics for naive bayes model"""
    # usage: ml_naive_bayes_metrics(naive_bayes_nm)
    y_true = naive_bayes_nm['y_test']
    y_pred = naive_bayes_nm['test_predictions']
    
    accuracy = sklmtrcs.accuracy_score(y_true, y_pred)
    precision = sklmtrcs.precision_score(y_true, y_pred, average='weighted')
    recall = sklmtrcs.recall_score(y_true, y_pred, average='weighted')
    f1 = sklmtrcs.f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def ml_naive_bayes_confusion(naive_bayes_nm):
    """create confusion matrix for naive bayes model"""
    # usage: ml_naive_bayes_confusion(naive_bayes_nm)
    y_true = naive_bayes_nm['y_test']
    y_pred = naive_bayes_nm['test_predictions']
    
    cm = sklmtrcs.confusion_matrix(y_true, y_pred)
    
    fig, ax = matplotlib.pyplot.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    matplotlib.pyplot.title('Confusion Matrix')
    matplotlib.pyplot.ylabel('True Label')
    matplotlib.pyplot.xlabel('Predicted Label')
    matplotlib.pyplot.show()
    
    return cm


def df_one_hot_enconding(df_name, col_name, *binary_bool:bool):
    """perform one-hot encoding on categorical variables"""
    # usage: df_one_hot_enconding(df, 'col_name', True) for binary encoding
    if binary_bool and binary_bool[0]:
        # binary encoding
        encoded_df = pd.get_dummies(df_name[col_name], prefix=col_name, drop_first=True)
    else:
        # full one-hot encoding
        encoded_df = pd.get_dummies(df_name[col_name], prefix=col_name)
    
    # combine with original dataframe
    result_df = pd.concat([df_name.drop(columns=[col_name]), encoded_df], axis=1)
    return result_df


def df_info_dtypes(df_name):
    """display dataframe info and data types"""
    # usage: df_info_dtypes(df)
    print("DataFrame Info:")
    df_name.info()
    print("\nData Types:")
    print(df_name.dtypes)
    return df_name.dtypes


def df_column_nms(df_name):
    """get column names of dataframe"""
    # usage: df_column_nms(df)
    return list(df_name.columns)