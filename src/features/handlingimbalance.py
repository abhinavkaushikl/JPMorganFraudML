import os
import mlflow
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from src.features.mlflowsetup import MLflowConfig
import pandas as pd
import mlflow
try:
    from src.dataloaders.load_data import DataLoader
    USE_DATALOADER = True
except (ImportError, FileNotFoundError) as e:
    print(f"Warning: Could not import DataLoader: {e}")
    USE_DATALOADER = False

class ImbalanceMonitor:
    def __init__(self):
        self.mlflow_setup = MLflowConfig()
        self.mlflow_setup.setup()
        
    def log_imbalance_stats(self, df: pd.DataFrame, target_col='Is_Fraud', run_name="imbalance_monitoring"):
        with mlflow.start_run(run_name=run_name):
            total_rows = len(df)
            mlflow.log_metric("total_rows", total_rows)
            
            # Get class distribution
            class_counts = Counter(df[target_col])
            
            # Log class distribution metrics
            for class_label, count in class_counts.items():
                safe_label = str(class_label).replace(" ", "_").replace(".", "_")
                percentage = count / total_rows * 100
                
                mlflow.log_metric(f"class_{safe_label}_count", count)
                mlflow.log_metric(f"class_{safe_label}_percentage", percentage)
            
            # Calculate imbalance ratio (majority class / minority class)
            majority_class = max(class_counts.items(), key=lambda x: x[1])
            minority_class = min(class_counts.items(), key=lambda x: x[1])
            
            imbalance_ratio = majority_class[1] / minority_class[1] if minority_class[1] > 0 else float('inf')
            mlflow.log_metric("imbalance_ratio", imbalance_ratio)
            
            # Log additional metrics
            num_classes = len(class_counts)
            mlflow.log_metric("num_classes", num_classes)
            
            # Calculate entropy as a measure of balance
            probabilities = np.array([count/total_rows for count in class_counts.values()])
            entropy = -np.sum(probabilities * np.log2(probabilities))
            max_entropy = np.log2(num_classes)  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            mlflow.log_metric("class_entropy", entropy)
            mlflow.log_metric("normalized_entropy", normalized_entropy)
            
            print(f"Logged imbalance stats for target column: {target_col}")
            print(f"Class distribution: {class_counts}")
            print(f"Imbalance ratio: {imbalance_ratio:.2f}")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
            print(f"Run URL: {mlflow.get_artifact_uri()}")
            
            # Return True if imbalance ratio exceeds a threshold (e.g., 10)
            is_imbalanced = imbalance_ratio > 10
            
            return is_imbalanced
    
    def apply_smote_if_needed(self, df: pd.DataFrame, target_col='Is_Fraud', threshold=10, 
                             processed_dir="/data/processed"):
        is_imbalanced = self.log_imbalance_stats(df, target_col)
        
        if is_imbalanced:
            print(f"Dataset is imbalanced. Applying SMOTE...")
            X_resampled, y_resampled = handle_imbalance_and_log(df, target_col, processed_dir)

            # # Log post-balancing metrics
            # with mlflow.start_run(run_name="post_balancing_metrics"):
            #     class_counts = Counter(balanced_df[target_col])
            #     for class_label, count in class_counts.items():
            #         safe_label = str(class_label).replace(" ", "_").replace(".", "_")
            #         percentage = count / len(balanced_df) * 100
                    
            #         mlflow.log_metric(f"balanced_class_{safe_label}_count", count)
            #         mlflow.log_metric(f"balanced_class_{safe_label}_percentage", percentage)
                
            #     print(f"After balancing - Class distribution: {class_counts}")
            
            return X_resampled,y_resampled
        else:
            print("Dataset is sufficiently balanced. No action needed.")
            return df


def handle_imbalance_and_log(df, target_col='Is_Fraud', processed_dir="/data/processed"):
    # Log original class distribution
    original_counts = Counter(df[target_col])

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Encode categorical columns using OneHotEncoder
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded_cat = encoder.fit_transform(X[categorical_cols])
        encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)
        df_encoded_cat = pd.DataFrame(X_encoded_cat, columns=encoded_cat_cols, index=X.index)
    else:
        df_encoded_cat = pd.DataFrame(index=X.index)

    # Combine numeric and encoded categorical columns
    X_processed = pd.concat([X[numeric_cols], df_encoded_cat], axis=1)

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    # # Ensure processed directory exists
    # os.makedirs(processed_dir, exist_ok=True)

    # # Save resampled X and y separately
    # X_output_path = os.path.join(processed_dir, "X_resampled.csv")
    # y_output_path = os.path.join(processed_dir, "y_resampled.csv")

    # pd.DataFrame(X_resampled, columns=X_processed.columns).to_csv(X_output_path, index=False)
    # pd.Series(y_resampled, name=target_col).to_csv(y_output_path, index=False)

    # print(f"X_resampled saved to {X_output_path}")
    # print(f"y_resampled saved to {y_output_path}")

    # # Optionally return a combined DataFrame if needed for downstream use
    # df_balanced = pd.concat([
    #     pd.DataFrame(X_resampled, columns=X_processed.columns),
    #     pd.Series(y_resampled, name=target_col)
    # ], axis=1)

    return X_resampled,y_resampled
