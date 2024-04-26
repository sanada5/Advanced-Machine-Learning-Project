# Regression Pipeline
 This repository contains a comprehensive machine learning pipeline for a regression task in the course ‘Advanced Machine Learning’ at ETH Zürich. The goal of this task is to apply data preprocessing techniques, hyperparameter tuning, model optimization to predict the age of the person from brain data (consisting of informative features derived from brain image data using FreeSurfer). The pipeline was developed by me and Erges Mema. 
# Key Features:
 ## Data Preprocessing:
   -Imputation of missing values using KNN imputation.
   
   -Outlier detection with the ECOD method.
   
   -Scaling of data using RobustScaler.
   
   -Feature selection based on F-statistic.
   
 ## Hyperparameter Tuning:
 ###  -Cross-validation and grid search for hyperparameter tuning of various regression models:
   -SVR (Support Vector Regressor)
   
   -RandomForestRegressor
   
   -GradientBoostingRegressor
   
   -DecisionTreeRegressor
   
   -AdaBoostRegressor
   
   -GaussianProcessRegressor
   
   -XGBRegressor
   
   -ExtraTreesRegressor
   
   -Ridge
   
   -ElasticNet
   
## Model Optimization:

   -StackingRegressor to combine predictions from multiple base estimators.
   
   -Bayesian optimization to find the best combination of base estimators for StackingRegressor.
   
   -Optimization of Gaussian Process Regressor kernel parameters to maximize R-squared score.
