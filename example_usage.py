# -*- coding: utf-8 -*-
"""
Example usage of the datashadric package
"""

import pandas as pd
import numpy as np

# this is how you'll import from your package once it's installed
# from datashadric.mlearning import ml_naive_bayes_model
# from datashadric.regression import lr_ols_model  
# from datashadric.dataframing import df_check_na_values
# from datashadric.stochastics import df_gaussian_checks
# from datashadric.plotters import df_boxplotter

def create_sample_data():
    """create sample data for demonstration"""
    np.random.seed(42)
    n = 1000
    
    # create sample dataset
    data = {
        'age': np.random.normal(35, 10, n),
        'income': np.random.normal(50000, 15000, n),
        'education': np.random.choice(['high_school', 'college', 'graduate'], n),
        'target': np.random.choice([0, 1], n),
        'measurement': np.random.normal(100, 15, n)
    }
    
    # add some missing values
    data['income'][np.random.choice(n, 50, replace=False)] = np.nan
    
    return pd.DataFrame(data)


def example_usage():
    """demonstrate package usage"""
    print("🔬 datashadric Package Example Usage")
    print("=" * 50)
    
    # create sample data
    df = create_sample_data()
    print(f"📊 Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    print("\n1. Data Quality Analysis:")
    # would use: na_summary = df_check_na_values(df)
    print("   - Check missing values with df_check_na_values()")
    print("   - Remove duplicates with df_drop_dupes()")
    
    print("\n2. Statistical Analysis:")
    # would use: normality = df_gaussian_checks(df, 'measurement')
    print("   - Test normality with df_gaussian_checks()")
    print("   - Calculate confidence intervals with df_calc_conf_interval()")
    
    print("\n3. Machine Learning:")
    # would use: model, metrics = ml_naive_bayes_model(df, 'target', test_size=0.2)
    print("   - Train Naive Bayes with ml_naive_bayes_model()")
    print("   - Evaluate models with ml_naive_bayes_metrics()")
    
    print("\n4. Regression Analysis:")
    # would use: results = lr_ols_model(df, 'income', ['age', 'target'])
    print("   - Run OLS regression with lr_ols_model()")
    print("   - Check assumptions with lr_check_homoscedasticity()")
    
    print("\n5. Visualization:")
    # would use: df_boxplotter(df, 'education', 'income', type_plot=1)
    print("   - Create box plots with df_boxplotter()")
    print("   - Generate histograms with df_histplotter()")
    
    print("\n📦 Installation Instructions:")
    print("   1. Navigate to the datashadric directory")
    print("   2. Run: pip install .")
    print("   3. Or for development: pip install -e .")
    print("   4. Then import: from datashadric.mlearning import ml_naive_bayes_model")


if __name__ == "__main__":
    example_usage()