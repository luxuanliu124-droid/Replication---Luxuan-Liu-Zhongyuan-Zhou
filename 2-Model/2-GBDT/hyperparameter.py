class live_config():
    '''
    store hyperparameter for GBDT, note that for gridsearch to run, each parameter should be stored in list.
    The choice of hyperparameters to tune concerning regularization. Ref:https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regularization.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regularization-py

    '''
    # Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
    n_estimators = [200]#[200, 300]
    # default lr. Learning rate shrinks the contribution of each tree
    learning_rate = [0.1]
    # maximum depth of the individual regression estimators (tune)
    max_depth = [3]#[3, 5, 7]
    # The fraction of features to consider when looking for the best split: int(max_features * n_features)
    max_features = [0.1]#[1.0, 0.5, 0.1]

    param_grid = {'n_estimators': n_estimators,
                  'learning_rate': learning_rate,
                  'max_depth': max_depth,
                  'max_features': max_features
                  }
    # filename of the GBDT model
    filename_gbdt = 'gbdt_best_estimator'

