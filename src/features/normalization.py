import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew
# Try to import the DataLoader, but fallback to direct file loading if it fails
try:
    from src.dataloaders.load_data import DataLoader
    USE_DATALOADER = True
except (ImportError, FileNotFoundError) as e:
    print(f"Warning: Could not import DataLoader: {e}")
    USE_DATALOADER = False

class Noramalization:
    def __init__(self, categorical_cols, target_col, save_path, save_format='csv'):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.save_path = save_path
        self.save_format = save_format
        self.label_encoders = {}

    def encode_categoricals(self, df):
        df_encoded = df.copy()

        for col in self.categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le

        print("Categorical encoding completed.")
        return df_encoded

    def normalize_numeric_features(self, df):
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target_col]

        for col in numeric_cols:
            data = df[col].dropna()
            sk = skew(data)

            # Choose scaler based on distribution
            if abs(sk) < 0.5:
                scaler = StandardScaler()
            elif abs(sk) > 2 or (data.std() / data.mean()) > 2:
                scaler = RobustScaler()
            else:
                scaler = MinMaxScaler()

            df_scaled[[col]] = scaler.fit_transform(df[[col]])

        print("Numeric scaling completed.")
        return df_scaled

    def preprocess(self, df):
        df_processed = self.encode_categoricals(df)
        df_processed = self.normalize_numeric_features(df_processed)
        return df_processed

    def save(self, df):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        if self.save_format == 'csv':
            df.to_csv(self.save_path, index=False)
        elif self.save_format == 'parquet':
            df.to_parquet(self.save_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {self.save_format}")

        print(f"Processed data saved to: {self.save_path}")

# # ========== Usage Example ==========
# if __name__ == "__main__":
#     # Paths and config
#     raw_path = "C:/Users/kau75421/LLMprojects/CreditFraudDetection/data/processed/Bank_Transaction_Fraud_Detection_Processed.csv"
#     #save_path = "bank_model_project/data/processed/processed_data.csv"

#     categorical_cols = [
#         'Gender', 'State', 'City', 'Account_Type',
#         'Transaction_Type', 'Merchant_Category', 'Transaction_Device',
#         'Transaction_Location', 'Device_Type', 'Transaction_Currency',
#         'Transaction_Description', 'Age_Group', 'User_Home_City',
#         'Customer_ID', 'Transaction_Date', 'Transaction_Time', 'Merchant_ID'
#     ]
#     target_col = 'Is_Fraud'

#     # Load raw data
#     try:
#         if USE_DATALOADER:
#             print("Attempting to load data using DataLoader...")
#             data_loader = DataLoader()
#             df = data_loader.load_data_processed()
#             print("Data loaded successfully using DataLoader")
#         else:
#             raise ImportError("DataLoader not available")
#     except Exception as e:
#         print(f"Error using DataLoader: {e}")
#         print("Falling back to direct file loading")
#         df = pd.read_csv(raw_path)
#         print(f"Data loaded from {raw_path}")

#     # Initialize and run pipeline
#     pipeline = Noramalization(
#         categorical_cols=categorical_cols,
#         target_col=target_col,
#         save_path=raw_path,
#         save_format='csv'  # or 'parquet'
#     )

#     df_processed = pipeline.preprocess(df)
#     pipeline.save(df_processed)
