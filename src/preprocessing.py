import pandas as pd
import numpy as np
from rulefit import RuleFit
from sklearn.exceptions import NotFittedError
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from pyod.models.ecod import ECOD


class Preprocessing:


  def __init__(self, imputer, ecod, knn_imputer, constant_filter, selector, scaler_x):

    self.imputer = imputer
    self.ecod = ecod
    self.knn_imputer = knn_imputer
    self.constant_filter = constant_filter
    self.selector = selector
    self.scaler_x = scaler_x
    self.outliers = None
    self.columns_to_drop = set()


  @staticmethod
  def load_data(url: str):

    df = pd.read_csv(url)
    return df

  def simple_imputation_and_outlier_ECOD(self, X, feature_names, y=None, fit_transformers=False):

    """
    Preprocesses the input data by imputing missing values, handling outliers,
    and performing outlier-aware imputation.

    Args:
        imputer: Imputer object for simple imputation.
        ecod: ECOD object for outlier detection.
        knn_imputer: Imputer object for KNN imputation.
        X: Input features.
        y: Target variable (optional).
        fit_transformers: Flag indicating whether to fit transformers.

    Returns:
        Preprocessed features and target variable (if provided).
    """

    if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
    if fit_transformers:
            X_imputed = self.imputer.fit_transform(X)
            # Convert the array back to a DataFrame and assign column names
            X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    else:
            X_imputed = self.imputer.transform(X)
            # Convert the array back to a DataFrame and assign column names
            X_imputed = pd.DataFrame(X_imputed, columns=X.columns)


    if fit_transformers:
        outliers = self.ecod.fit_predict(X_imputed)
        X, y = X[outliers == 0], y[outliers == 0]


    if fit_transformers:
            X_imputed = self.knn_imputer.fit_transform(X)
    else:
            X_imputed = self.knn_imputer.transform(X)


    return (X_imputed, y) if y is not None else X_imputed


  def feature_selection(self, X, y=None, fit_transformers=False):
    
    """
    Perform feature selection by removing constant value columns, highly correlated features, and selecting k best features.

    Args:
        constant_filter: Filter to remove constant value columns.
        selector: Selector object for selecting k best features.
        columns_to_drop: Set to store indices of columns to be dropped.
        X: Input features.
        y: Target variable (optional).
        fit_transformers: Flag indicating whether to fit transformers.

    Returns:
        Transformed features and target variable (if provided).
    """
    
  
    # Remove constant value columns
    # if not isinstance(X, pd.DataFrame):
    #         X = pd.DataFrame(X)
    # if not isinstance(y, pd.DataFrame):
    #         y = pd.DataFrame(y)
    if fit_transformers:
        X = self.constant_filter.fit_transform(X)
    else:
        X = self.constant_filter.transform(X)

    # Remove highly correlated features
    # Assume X is a numpy array and y is a numpy array or pandas Series
    #   Calculate the correlation matrix from X
    corr_matrix = np.corrcoef(X, rowvar=False)  # Compute correlation between columns
    corr_matrix = np.abs(corr_matrix)  # Take the absolute value of correlation coefficients

    # Find indices of highly correlated features (above 0.8 threshold)
    high_corr_var_indices = np.where(corr_matrix > 0.8)

    # Filter out the upper triangle and diagonal elements to avoid duplicate pairs and self-correlation
    high_corr_var_pairs = [(i, j) for i, j in zip(*high_corr_var_indices) if i < j]

    if fit_transformers==True:
        # Iterate over each pair and drop the one that is less correlated with y
        for (index1, index2) in high_corr_var_pairs:
            if np.corrcoef(X[:, index1], y)[0, 1] < np.corrcoef(X[:, index2], y)[0, 1]:
                self.columns_to_drop.add(index1)
            else:
                self.columns_to_drop.add(index2)
        # for (index1, index2) in high_corr_var_pairs:
        #     x1 = X[:, index1:index1+1]  # Keep it 2D
        #     x2 = X[:, index2:index2+1]  # Keep it 2D
        #
        #     if np.corrcoef(x1.flatten(), y)[0, 1] < np.corrcoef(x2.flatten(), y)[0, 1]:
        #         columns_to_drop.add(index1)
        #     else:
        #         columns_to_drop.add(index2)
        X = np.delete(X, list(self.columns_to_drop), axis=1)
    else:
        # Drop the columns from X
        X = np.delete(X, list(self.columns_to_drop), axis=1)

    # SelectKBest for remaining features
    if fit_transformers:
        X_new = self.selector.fit_transform(X, y)
    else:
        X_new = self.selector.transform(X)
    return (X_new, y) if y is not None else X_new

  def scale_data(self, X, y=None, fit_transformers=False):

    """
    Scale the input features and target variable (if provided) using a scaler.

    Args:
        X: Input features.
        y: Target variable (optional).
        fit_transformers: Flag indicating whether to fit transformers.

    Returns:
        Scaled features and target variable (if provided).
    """

    if fit_transformers:
        X_transformed = self.scaler_x.fit_transform(X)

    else:
        X_transformed = self.scaler_x.transform(X)

    return (X_transformed, y) if y is not None else X_transformed

  

