import json

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from data_reading import read_df


def grid_search(config):
    df, feature_names = read_df(config)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[feature_names],
                                                        df['one_if_male'],
                                                        test_size=config['test_size'],
                                                        random_state=config['random_state'])

    # Define the undersampling strategy
    undersampler = RandomUnderSampler(random_state=config['random_state'])

    # Resample the training data
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    # Resample the test data
    X_test_resampled, y_test_resampled = undersampler.fit_resample(X_test, y_test)

    print(f'X_train: {X_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_train: {np.mean(y_train)}')
    print(f'y_test: {np.mean(y_test)}')

    print()
    print(f'X_train_resampled: {X_train_resampled.shape}')
    print(f'X_test_resampled: {X_test_resampled.shape}')
    print(f'y_train_resampled: {np.mean(y_train_resampled)}')
    print(f'y_test_resampled: {np.mean(y_test_resampled)}')

    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=config['max_iter'], verbose=3)

    # Perform grid search
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)
    grid_search.fit(X_train_resampled, y_train_resampled)
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score (accuracy): {:.2f}".format(grid_search.best_score_))
