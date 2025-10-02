# -*- coding: utf-8 -*-
"""
Plotters Functions Module
Comprehensive collection of plotting and visualization utilities for data exploration and presentation
"""

# third-party data science imports
import pandas as pd
import numpy as np

# visualization imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def df_boxplotter(df_name, col_xplot, col_yplot, type_plot: int, *args):
    """create box plot to visualize outliers. type_plot: 0 for dist, 1 for money, 2 for general"""
    # usage: df_boxplotter(df, 'col_x', 'col_y', type_plot=0, 'horizontalalignment')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sns.boxplot(x=df_name[col_xplot], y=df_name[col_yplot], ax=ax)
    plt.title('{} box plot to visualise outliers'.format(col_yplot))
    
    if type_plot == 0:
        plt.ylabel('{} in miles'.format(col_yplot))
    elif type_plot == 1:
        plt.ylabel('{} in $'.format(col_yplot))
    else:
        plt.ylabel('{}'.format(col_yplot))
    
    if args:
        plt.xticks(rotation=0, horizontalalignment=args[0])
    
    ax.yaxis.grid(True)
    plt.savefig("Boxplot_x-{}_y-{}.png".format(col_xplot, col_yplot))
    plt.show()


def df_histplotter(df_name, col_plot, type_plot: int, bins=10, *args):
    """create histogram plot. type_plot: 0 for dist, 1 for money"""
    # usage: df_histplotter(df, 'col_name', type_plot=0, bins=20)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    df_name[col_plot].hist(bins=bins, ax=ax)
    plt.title('{} histogram'.format(col_plot))
    
    if type_plot == 0:
        plt.xlabel('{} in miles'.format(col_plot))
    elif type_plot == 1:
        plt.xlabel('{} in $'.format(col_plot))
    else:
        plt.xlabel('{}'.format(col_plot))
    
    plt.ylabel('Frequency')
    ax.grid(True)
    plt.savefig("Histogram_{}.png".format(col_plot))
    plt.show()


def df_grouped_histplotter(df_name, col_groupby: str, col_plot: str, type_plot: int, bins=20):
    """create grouped histogram plots"""
    # usage: df_grouped_histplotter(df, 'col_groupby', 'col_plot', type_plot=0, bins=20)
    groups = df_name.groupby(col_groupby)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    for name, group in groups:
        group[col_plot].hist(bins=bins, alpha=0.7, label=name, ax=ax)
    
    plt.title('{} histogram grouped by {}'.format(col_plot, col_groupby))
    plt.xlabel(col_plot)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def df_grouped_barplotter(df_name, col_groupby: str, col_plot: str, type_plot: int):
    """create grouped bar plots"""
    # usage: df_grouped_barplotter(df, 'col_groupby', 'col_plot', type_plot=0)
    grouped_data = df_name.groupby(col_groupby)[col_plot].mean()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    grouped_data.plot(kind='bar', ax=ax)
    plt.title('{} by {}'.format(col_plot, col_groupby))
    plt.xlabel(col_groupby)
    plt.ylabel(col_plot)
    plt.xticks(rotation=45)
    plt.show()


def df_scatterplotter(df_grouped, col_xplot, col_yplot):
    """create scatter plot between two variables"""
    # usage: df_scatterplotter(df, 'col_x', 'col_y')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    df_grouped.plot.scatter(x=col_xplot, y=col_yplot, ax=ax)
    plt.title('Scatter plot: {} vs {}'.format(col_xplot, col_yplot))
    plt.show()


def df_pairplot(data):
    """create pairplot for data exploration"""
    # usage: df_pairplot(df)
    sns.pairplot(data)
    plt.show()