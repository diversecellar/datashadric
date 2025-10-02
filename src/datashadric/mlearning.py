# -*- coding: utf-8 -*-
"""
Machine Learning Functions Module
Comprehensive collection of machine learning utilities for model training, evaluation, and prediction
"""

# third-party machine learning imports
import sklearn.linear_model as skllinmod
import sklearn.naive_bayes as sklnvbys
import sklearn.metrics as sklmtrcs
import sklearn.model_selection as sklmodslct

# third-party data science imports
import pandas as pd
import numpy as np

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns


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


def logr_train_test_split(df_name, col_response, col_predictor, test_size: float):
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


def ml_train_test_split(df_name, col_target, test_size: float):
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return cm