from src.train import Training

class Prediction:
    
    def __init__(self, train_):
        self.train_ = train_

    def generate_predictions(self, X_train, y_train, X_test):
        self.train_.train(X_train, y_train)
        y_pred = self.train_.predict(X_test)
        return y_pred


    def resize(self, scaler, y_pred):

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        # Inverse transform the scaled output
        y_pred = scaler_y.inverse_transform(y_pred)
        return y_pred

        
        

        