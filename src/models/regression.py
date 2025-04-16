from sklearn.tree import DecisionTreeRegressor
from src.models.train_split import get_data
import joblib
import os
import warnings

class RegressionModel:
    """
    A custom Regession model to get predicts for train and test.
    ---
    """
    model = DecisionTreeRegressor()
        
    def fit(self, verbose=False):
        x_train, x_test, y_train, y_test, self.label_encoder = get_data(split_type="regression", verbose=verbose)
        self.model.fit(x_train, y_train)
        
        return x_train, x_test, y_train, y_test
    
    def get_all_preds(self):
        x_train, x_test, y_train, y_test = self.fit()
        train_pred = self.model.predict(x_train)
        test_pred = self.model.predict(x_test)
        return train_pred, y_train, test_pred, y_test

    def get_train_pred(self):
        train_pred, y_train, _, _ = self.get_all_preds()
        return train_pred, y_train
    
    def get_test_pred(self):
        _, _, test_pred, y_test = self.get_all_preds()
        return test_pred, y_test
    
    def save(self, root:str="weigths"):
        path = os.path.join(root, "m_classify.joblib")
        if os.path.exists:
           warnings.warn(f"Model is overiding to because the same dir, the same model name is exist") 
           return
        joblib.dump(self.model, path)
        
        print(f"Model saved in {path}")
    
    def load(self, path:str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        
    def __repr__(self):
        return f"sklearn.DecisionTreeRegressor()"
