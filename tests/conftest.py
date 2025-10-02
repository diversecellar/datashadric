# -*- coding: utf-8 -*-
"""
Test configuration for datashadric package
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_dataframe():
    """create a sample dataframe for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric_col': np.random.normal(100, 15, 1000),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 1000),
        'binary_col': np.random.choice([0, 1], 1000),
        'with_nulls': np.random.choice([1, 2, 3, np.nan], 1000),
    })


@pytest.fixture
def regression_data():
    """create sample data for regression testing"""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 2 + 3*x1 + 1.5*x2 + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })