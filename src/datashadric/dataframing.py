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
    # usage: df_print_row_and_columns(df)
    # input: df_name - pandas DataFrame
    # output: prints number of rows and columns to console
    try:
        df_rows, df_columns = df_name.shape
    except Exception as e:
        df_rows, df_columns = df_name.to_frame().shape
    print("rows = {}".format(df_rows))
    print("columns = {}".format(df_columns))


def df_check_na_values(df_name, *args):
    """check for missing values in dataframe columns"""
    # usage: df_check_na_values(df) or df_check_na_values(df, ['col1', 'col2'])
    # input: df_name - pandas DataFrame, args - optional list of column names to check for missing values
    # output: boolean DataFrame indicating missing values
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
    # input: df_name - pandas DataFrame, col_datetime_lookup - substring to identify date columns
    # output: DataFrame with specified columns converted to datetime format
    for column in df_name.columns.tolist():
        if str(col_datetime_lookup) in str(column):
            print("yes")
            df_name[column] = pd.to_datetime(df_name[column])

    return df_name


def df_explore_unique_categories(df_name, col):
    """print a dataframe with unique categories for each categorical variable"""
    # usage: df_explore_unique_categories(df, 'col_name')
    # input: df_name - pandas DataFrame, col - column name to explore
    # output: DataFrame with unique values in specified column
    df_col_unique = df_name.drop_duplicates(subset=col, keep='first')

    return df_col_unique[col]


def df_mask_with_list(df, df_col, list_comp: list, mask_type: int):
    """mask dataframe with list comparison. mask_type: 0 for isin, 1 for not isin"""
    # usage: df_mask_with_list(df, 'col_name', ['val1', 'val2'], mask_type=0)
    # input: df - pandas DataFrame, df_col - column name to apply mask on, list_comp - list of values for comparison, mask_type - 0 for isin, 1 for not isin
    # output: masked DataFrame
    if mask_type == 0:
        mask = df[df_col].isin(list_comp)
    else:
        mask = ~df[df_col].isin(list_comp)

    return df[mask]


def df_groupby_mask_operate(df, col_name_masker: str, col_name_operate: str, *args):
    """group by and perform operations on masked data"""
    # usage: df_groupby_mask_operate(df, 'col_groupby', 'col_operate', 'mean')
    # input: df - pandas DataFrame, col_name_masker - column name to group by, col_name_operate - column name to perform operation on, args - operation to perform (e.g., 'mean', 'sum')
    # output: DataFrame with grouped statistics
    grouped = df.groupby(col_name_masker)[col_name_operate]
    if args:
        return grouped.agg(args[0])
    
    return grouped.describe()


def df_cross_corr_check(df_name, cols_y: list, cols_x: list):
    """check cross-correlation between y and x variables"""
    # usage: df_cross_corr_check(df, ['col_y1', 'col_y2'], ['col_x1', 'col_x2'])
    # input: df_name - pandas DataFrame, cols_y - list of y variable column names, cols_x - list of x variable column names
    # output: DataFrame with cross-correlation matrix
    correlation_matrix = df_name[cols_y + cols_x].corr()

    return correlation_matrix.loc[cols_y, cols_x]


def df_class_balance(df_filtered):
    """check class balance in filtered dataframe"""
    # usage: df_class_balance(df_filtered)
    # input: df_filtered - pandas DataFrame with categorical variables
    # output: DataFrame with counts and percentages of each class
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
    # input: df - pandas DataFrame, col_dupes - 0 to drop duplicates based on all columns, 1 to drop based on specified columns in args
    # output: DataFrame with duplicates removed
    if args:
        subset_cols = args[0] if isinstance(args[0], list) else [args[0]]
        return df.drop_duplicates(subset=subset_cols, keep='first')
    
    return df.drop_duplicates(keep='first')


def df_drop_col(df, col_name: str):
    """drop specified column from dataframe"""
    # usage: df_drop_col(df, 'col_name')
    # input: df - pandas DataFrame, col_name - column name to drop
    # output: DataFrame with specified column removed
    if col_name in df.columns:
        print(f"Dropping Column: {col_name} at index {df.columns.get_loc(col_name)}")
        return df.drop(columns=[col_name])
    
    return df


def df_drop_multicol(df, col_names: list):
    """drop specified columns from dataframe"""
    # usage: df_drop_multicol(df, ['col1', 'col2'])
    # input: df - pandas DataFrame, col_names - list of column names to drop
    # output: DataFrame with specified columns removed
    cols_to_drop = [col for col in col_names if col in df.columns]
    if cols_to_drop:
        for col_name in cols_to_drop:
            print(f"Dropping Column: {col_name} at index {df.columns.get_loc(col_name)}")
        return df.drop(columns=cols_to_drop)

    return df
 

def df_corr_check(df_name, col_y, col_x):
    """check correlation between two variables"""
    # usage: df_corr_check(df, 'col_y', 'col_x')
    # input: df_name - pandas DataFrame, col_y - first variable column name, col_x - second variable column name
    # output: correlation coefficient between the two variables
    correlation = df_name[[col_y, col_x]].corr().iloc[0, 1]

    return correlation


def df_head(df_name, head_num: int):
    """display first n rows of dataframe"""
    # usage: df_head(df, head_num=5)
    # input: df_name - pandas DataFrame, head_num - number of rows to display
    # output: first n rows of the DataFrame
    return df_name.head(head_num)


def df_one_hot_enconding(df_name, col_name, *binary_bool: bool):
    """perform one-hot encoding on categorical variables"""
    # usage: df_one_hot_enconding(df, 'col_name', True) for binary encoding
    # input: df_name - pandas DataFrame, col_name - column name to encode, binary_bool - optional boolean for binary encoding (True for binary, False for full one-hot)
    # output: DataFrame with one-hot encoded columns
    if binary_bool and binary_bool[0]:
        # binary encoding
        encoded_df = pd.get_dummies(df_name[col_name], prefix=col_name, drop_first=True)
    else:
        # full one-hot encoding
        encoded_df = pd.get_dummies(df_name[col_name], prefix=col_name)
    
    # combine with original dataframe
    result_df = pd.concat([df_name.drop(columns=[col_name]), encoded_df], axis=1)

    return result_df


def df_info_dtypes(df_name, *args):
    """display dataframe info and data types"""
    # usage: df_info_dtypes(df, "v"") or df_info_dtypes(df) "v for detailed info"
    # input: df_name - pandas DataFrame
    # output: prints dataframe info and data types to console, still returns data types for further use
    if args:
        if str(args[0])[:1].lower() == "v":
            print("Detailed DataFrame Info:")
            print(df_name.info(verbose=True, null_counts=True))
            print("\nDataFrame Data Types:")
            print(df_name.dtypes)
        else:
            print("\033[91mNB: Only valid arg is 'v' for verbose info\033[0m")
            print("\033[96mNB: Dataframe Data Types still returned\033[0m")

    return df_name.dtypes


def df_column_nms(df_name, *args):
    """get column names of dataframe"""
    # usage: df_column_nms(df, "v") or df_column_nms(df) "v to print columns"
    # input: df_name - pandas DataFrame
    # output: list of column names, print to console
    if args: 
        if str(args[0])[:1].lower() == "v":
            print("DataFrame Columns:")
            print(list(df_name.columns))
        else:
            print("\033[91mNB: Only valid arg is 'v' for verbose info\033[0m")
            print("\033[96mNB: Dataframe Columns still returned\033[0m")

    return list(df_name.columns)


def remove_whitespace(str_target: str):
    """remove whitespace from string"""
    # usage: remove_whitespace(' some text ')
    # input: str_target - input string
    # output: string with whitespace removed
    return str_target.replace(' ', '')


def remove_unicode(str_target: str):
    """remove unicode characters from string"""
    # usage: remove_unicode('café')
    # input: str_target - input string
    # output: string with unicode characters replaced by closest ASCII equivalent
    try:
        clean_string = unidecode.unidecode(str_target)
        return clean_string
    except Exception as e:
        print(f"Error cleaning unicode: {e}")
        return str_target
    

def degree_symbol_parse(str_target: str):
    """replace '90deg' with '90°'"""
    # usage: degree_symbol_parse('Turn 90deg to the right')
    # input: str_target - input string
    # output: string with 'deg' replaced by '°'
    clean_string = re.sub(r'(\d+)deg\b', r'\1°', str_target)

    return clean_string