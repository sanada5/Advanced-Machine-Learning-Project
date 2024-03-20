from rulefit import RuleFit
from sklearn.exceptions import NotFittedError
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNet, Ridge
from pyod.models.ecod import ECOD
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor, GradientBoostingRegressor, \
    StackingRegressor, AdaBoostRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import QuantileTransformer, StandardScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, WhiteKernel
from sklearn.gaussian_process.kernels import RBF

"""
Initialize machine learning models for use in a machine learning pipeline.

Models are instantiated with predefined hyperparameters for ease of use in
training and prediction tasks.

Models included:
- Support Vector Regressor (SVR)
- Random Forest Regressor (RFR)
- Gradient Boosting Regressor (GB)
- Decision Tree Regressor (DTR)
- AdaBoost Regressor (AB)
- Gaussian Process Regressor (GPR)
- XGBoost Regressor (XGB)
- Extra Trees Regressor (ET)
- Ridge Regression (Ridge)
- ElasticNet Regression (ElasticNet)

Parameters:
-----------
None

Returns:
--------
svr : SVR
    Support Vector Regressor with predefined hyperparameters.
rfr : RandomForestRegressor
    Random Forest Regressor with predefined hyperparameters.
gb : GradientBoostingRegressor
    Gradient Boosting Regressor with predefined hyperparameters.
dtr : DecisionTreeRegressor
    Decision Tree Regressor with predefined hyperparameters.
ab : AdaBoostRegressor
    AdaBoost Regressor with predefined hyperparameters.
gpr : GaussianProcessRegressor
    Gaussian Process Regressor with predefined hyperparameters.
xgb : XGBRegressor
    XGBoost Regressor with predefined hyperparameters.
et : ExtraTreesRegressor
    Extra Trees Regressor with predefined hyperparameters.
ridge : Ridge
    Ridge Regression with predefined hyperparameters.
elasticnet : ElasticNet
    ElasticNet Regression with predefined hyperparameters.
"""


# Instantiating the modelssvr = SVR(C=10, epsilon=0.1, gamma=0.01, kernel='rbf')

svr = SVR(C=10, epsilon=0.1, gamma=0.01, kernel='rbf')

rfr = RandomForestRegressor(random_state=0, max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=500)

gb = GradientBoostingRegressor(learning_rate= 0.01, max_depth= 5, n_estimators= 1000, subsample= 0.5)

dtr = DecisionTreeRegressor(max_depth=3, min_samples_leaf=4, min_samples_split=2, random_state=42)

ab = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), learning_rate=0.1, loss='exponential', n_estimators=500, random_state=42)

gpr = GaussianProcessRegressor(alpha=0.01, kernel=(1**2 * RationalQuadratic(alpha=0.1, length_scale=1)),random_state=0)

xgb = XGBRegressor(gamma=0, learning_rate = 0.01, max_depth = 5, n_estimators = 1000, random_state=0, subsample=0.5)

et = ExtraTreesRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=500,random_state=42)

# lgbm = LGBMRegressor(n_estimators=100, random_state=42, learning_rate=0.1, num_leaves=31, boosting_type='gbdt')

# cb = CatBoostRegressor(iterations=1000,depth=6,learning_rate=0.03, random_state=42, l2_leaf_reg=3, silent=True)

ridge = Ridge(alpha=100)

#rvr = EMRVR(kernel='rbf')

elasticnet = ElasticNet(l1_ratio=0.1, alpha=0.1746603642227994)