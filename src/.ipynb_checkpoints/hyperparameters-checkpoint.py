from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, WhiteKernel
from sklearn.gaussian_process.kernels import RBF

SVR_PARAMS = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly'],
    'epsilon': [0.1, 0.2, 0.5, 0.3]
}

RFR_PARAMS = {
    'n_estimators': [10, 50, 100, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

GB_PARAMS = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.001, 0.01, 0.1],
    'subsample': [0.5, 0.7, 1.0],
    'max_depth': [3, 4, 5]
}

DTR_PARAMS = {
    'max_depth': [1, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

AB_PARAMS = {
    'base_estimator__max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.001, 0.01, 0.1],
    'loss': ['linear', 'square', 'exponential'],
    'random_state': [42]
}

GPR_PARAMS = {
    'alpha': [1e-10, 1e-2, 1, 100],
    'kernel': [1**2 * RBF(length_scale=1), 1**2 * RationalQuadratic(alpha=0.1, length_scale=1)],
    'random_state': [0]
}

XGB_PARAMS = {
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.001, 0.01, 0.1],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.5, 0.7, 1.0],
    'random_state': [0]
}

ET_PARAMS = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42]
}

RIDGE_PARAMS = {
    'alpha': [0.1, 1.0, 10.0, 12, 100.0]
}

ELASTICNET_PARAMS = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}

