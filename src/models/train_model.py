import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Reset to default tracking URI
mlflow.set_tracking_uri(None)
print("Using default MLflow tracking URI")

class ModelTrainer:
    def __init__(self, X, y, model_type):
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=['int']).columns:
                X[col] = X[col].astype(float)
        self.X = X
        self.y = y
        self.model_type = model_type
        self.model = None

    def train_random_forest(self):
        print("Training Random Forest on full data...")
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self.model.fit(self.X, self.y)

    def train_lightgbm(self):
        print("Training LightGBM on full data...")
        self.model = lgb.LGBMClassifier(objective='binary', is_unbalance=False, random_state=42)
        self.model.fit(self.X, self.y)

    def train(self):
        if self.model_type == "rf":
            self.train_random_forest()
        elif self.model_type == "lgbm":
            self.train_lightgbm()
        else:
            raise ValueError("Unsupported model type")
        return self.model


def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")


def load_data(path, target_col):
    df = pd.read_csv(path)
    if 'Bank_Branch' in df.columns:
        df.drop(columns='Bank_Branch', inplace=True)
    return df.drop(columns=[target_col]), df[target_col]

