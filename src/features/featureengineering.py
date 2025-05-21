import pandas as pd
import numpy as np
import os
from pathlib import Path

# Try to import the DataLoader, but fallback to direct file loading if it fails
try:
    from src.dataloaders.load_data import DataLoader
    USE_DATALOADER = True
except (ImportError, FileNotFoundError) as e:
    print(f"Warning: Could not import DataLoader: {e}")
    USE_DATALOADER = False

class FeatureEngineer:
    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns if drop_columns else [
            'Customer_Name', 'Transaction_ID', 'Customer_Contact', 'Customer_Email'
        ]
        self.user_home_city_map = None

    def _drop_columns(self, df):
        return df.drop(columns=self.drop_columns, errors='ignore')

    def _temporal_features(self, df):
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
        df['Transaction_Hour'] = df['Transaction_Time'].astype(str).str.zfill(4).str[:2].astype(int)
        df['HourOfDay'] = df['Transaction_Hour']
        df['Is_Night'] = df['HourOfDay'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
        df['Is_Weekend'] = df['Transaction_Date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        return df

    def _money_features(self, df):
        df['Balance_After_Transaction'] = df['Account_Balance'] - df['Transaction_Amount']
        df['Amount_Per_Balance'] = df.apply(
            lambda row: row['Transaction_Amount'] / row['Account_Balance'] if row['Account_Balance'] > 0 else 0,
            axis=1
        )
        threshold = df['Transaction_Amount'].quantile(0.95)
        df['Is_Large_Transaction'] = (df['Transaction_Amount'] > threshold).astype(int)
        if 'Customer_ID' in df.columns:
            df['Daily_Spend'] = df.groupby(['Customer_ID', df['Transaction_Date'].dt.date])['Transaction_Amount'].transform('sum')
        return df

    def _age_features(self, df):
        def get_age_group(age):
            if age < 25:
                return 'Young'
            elif age <= 60:
                return 'Adult'
            else:
                return 'Senior'

        df['Age_Group'] = df['Age'].apply(get_age_group)
        df['Is_Senior'] = (df['Age'] > 60).astype(int)
        df['Is_Youth'] = (df['Age'] < 25).astype(int)
        return df

    def _device_features(self, df):
        df['Transaction_Hour'] = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S', errors='coerce').dt.hour

        def detect_device_hour_anomaly(row):
            if row['Transaction_Device'] == 'POS' and (row['Transaction_Hour'] < 5 or row['Transaction_Hour'] > 23):
                return 1
            if row['Transaction_Device'] == 'ATM' and row['Transaction_Hour'] < 4:
                return 1
            return 0

        df['Device_Hour_Anomaly'] = df.apply(detect_device_hour_anomaly, axis=1)
        return df

    def _location_features(self, df):
        if self.user_home_city_map is None and 'Customer_ID' in df.columns and 'City' in df.columns:
            self.user_home_city_map = df.groupby('Customer_ID')['City'].agg(lambda x: x.mode().iloc[0])

        if self.user_home_city_map is not None:
            df['User_Home_City'] = df['Customer_ID'].map(self.user_home_city_map)
            df['Device_Location_Mismatch'] = (
                (df['Transaction_Location'] != df['User_Home_City']) |
                (df['City'] != df['User_Home_City'])
            ).astype(int)
        return df

    def transform(self, df):
        df = df.copy()
        df = self._drop_columns(df)
        df = self._temporal_features(df)
        df = self._money_features(df)
        df = self._age_features(df)
        df = self._device_features(df)
        df = self._location_features(df)
        return df

    def save(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)


# if __name__ == "__main__":
#     # Define paths for fallback
#     project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from this file
#     raw_data_path = project_root / "data" / "raw" / "Bank_transaction_Fraud_Detection.csv"
#     output_path = project_root / "data" / "processed" / "Bank_Transaction_Fraud_Detection_Processed.csv"
    
#     # Load data
#     try:
#         if USE_DATALOADER:
#             print("Attempting to load data using DataLoader...")
#             data_loader = DataLoader()
#             df = data_loader.load_data_raw()
#             print("Data loaded successfully using DataLoader")
#         else:
#             raise ImportError("DataLoader not available")
#     except Exception as e:
#         print(f"Error using DataLoader: {e}")
#         print("Falling back to direct file loading")
#         df = pd.read_csv(raw_data_path)
#         print(f"Data loaded from {raw_data_path}")
    
#     # Process data
#     fe = FeatureEngineer()
#     df_processed = fe.transform(df)
#     fe.save(df_processed, output_path)
#     print(f"Processed data saved to {output_path}")
