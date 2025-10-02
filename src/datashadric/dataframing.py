# -*- coding: utf-8 -*-
"""
Dataframing Functions Module
Comprehensive collection of pandas and numpy utilities for data manipulation and preprocessing
"""

# standard library imports
import re

# third-party data science imports
import pandas as pd
import numpy as np
import unidecode


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


def df_explore_unique_categories(df_name, col):
    """print a dataframe with unique categories for each categorical variable"""
    # usage: df_explore_unique_categories(df, 'col_name')
    df_col_unique = df_name.drop_duplicates(subset=col, keep='first')
    return df_col_unique[col]


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


def df_corr_check(df_name, col_y, col_x):
    """check correlation between two variables"""
    # usage: df_corr_check(df, 'col_y', 'col_x')
    correlation = df_name[[col_y, col_x]].corr().iloc[0, 1]
    return correlation


def df_head(df_name, head_num: int):
    """display first n rows of dataframe"""
    # usage: df_head(df, head_num=5)
    return df_name.head(head_num)


def df_one_hot_enconding(df_name, col_name, *binary_bool: bool):
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