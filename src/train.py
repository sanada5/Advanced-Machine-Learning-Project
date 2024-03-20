from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor, GradientBoostingRegressor, \
    StackingRegressor, AdaBoostRegressor, VotingRegressor, ExtraTreesRegressor

class Training:

  def __init__(self, estimator, param_grid=None):


    self.estimator = estimator
    self.param_grid = param_grid

  def train(self, X_train, y_train):

      
       
       self.estimator.fit(X_train, y_train)

  def find_best_parameters(self, X_train, y_train):
      
    grid_search = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, cv=5,   
                               scoring='r2', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    # y_pred = best_estimator.predict(X_test)
    # r2 = r2_score(y_test, y_pred)

    print(f'Best parameters for {self.estimator.__class__.__name__}:{grid_search.best_params_}')
    print(f'R^2 score: {best_score}')
 

   

  