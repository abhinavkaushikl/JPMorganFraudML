from src.features.mlflowsetup import MLflowConfig
import pandas as pd
import mlflow
from src.features.mlflowsetup import MLflowConfig
from sklearn.experimental import enable_iterative_imputer  # noqa: F401from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
try:
    from src.dataloaders.load_data import DataLoader
    USE_DATALOADER = True
except (ImportError, FileNotFoundError) as e:
    print(f"Warning: Could not import DataLoader: {e}")
    USE_DATALOADER = False

class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.df_cleaned = df.copy()

    def remove_high_null_columns(self, threshold=70):
        try:
            null_percentage = self.df_cleaned.isnull().mean() * 100
            cols_to_remove = null_percentage[null_percentage > threshold].index
            self.df_cleaned.drop(columns=cols_to_remove, inplace=True)
            print(f"Removed columns with more than {threshold}% missing values: {cols_to_remove.tolist()}")
        except Exception as e:
            print(f"Error in removing high null columns: {e}")

    def impute_numerical_values(self):
        try:
           
            numerical_cols = self.df_cleaned.select_dtypes(include=['number']).columns
            skewness = self.df_cleaned[numerical_cols].skew()

            symmetric_columns = skewness[(skewness >= -0.5) & (skewness <= 0.5)].index.tolist()
            skewed_columns = skewness[skewness.abs() > 1].index.tolist()

            if symmetric_columns:
                imputer_mean = SimpleImputer(strategy='mean')
                self.df_cleaned[symmetric_columns] = imputer_mean.fit_transform(self.df_cleaned[symmetric_columns])

            if skewed_columns:
                imputer_median = SimpleImputer(strategy='median')
                self.df_cleaned[skewed_columns] = imputer_median.fit_transform(self.df_cleaned[skewed_columns])

            print("Numerical imputation complete.")
        except Exception as e:
            print(f"Error in numerical imputation: {e}")
    def impute_categorical_values(self):
        try:
            categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                mode = self.df_cleaned[col].mode()
                if not mode.empty:
                    self.df_cleaned[col] = self.df_cleaned[col].fillna(mode[0])
            print("Categorical imputation complete.")
        except Exception as e:
            print(f"Error in categorical imputation: {e}")

    def find_columns_with_nulls(self):
        try:
            null_percentage = self.df_cleaned.isnull().mean() * 100
            return null_percentage[null_percentage > 0].index.tolist()
        except Exception as e:
            print(f"Error finding columns with nulls: {e}")
            return []

    def impute_with_iterative(self, columns):
        try:
          
            imputer = IterativeImputer(random_state=42)
            self.df_cleaned[columns] = imputer.fit_transform(self.df_cleaned[columns])
            print("Iterative imputation completed.")
        except Exception as e:
            print(f"Error in iterative imputation: {e}")


class NullValueMonitor:
    def __init__(self):
        self.mlflow_setup = MLflowConfig()
        self.mlflow_setup.setup()

    def log_null_stats(self, df: pd.DataFrame, processor: DataProcessor = None, run_name="null_monitoring_new"):
        with mlflow.start_run(run_name=run_name):
            total_rows = len(df)
            mlflow.log_metric("total_rows", total_rows)

            null_found = False

            for col in df.columns:
                null_count = df[col].isnull().sum()
                null_percent = null_count / total_rows if total_rows > 0 else 0

                safe_col = col.replace(" ", "_").replace(".", "_")
                mlflow.log_metric(f"Null_count_{safe_col}", null_count)
                mlflow.log_metric(f"Null_percent_{safe_col}", null_percent)

                if null_count > 0:
                    null_found = True

            print(f"Logged null stats for {len(df.columns)} columns.")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
            print(f"Run URL: {mlflow.get_artifact_uri()}")

            if null_found and processor:
                print("Null values found. Running cleaning pipeline...\n")
                processor.remove_high_null_columns(threshold=70)
                processor.impute_numerical_values()
                processor.impute_categorical_values()
                columns_with_nulls = processor.find_columns_with_nulls()
                if columns_with_nulls:
                    processor.impute_with_iterative(columns_with_nulls)
            else:
                print("No nulls found. Skipping cleaning.")

            return null_found


# # --- Usage Example ---
# if __name__ == "__main__":
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
#         # Note: raw_path is not defined in the original code
#         # You might want to define a default path here
#         #raw_path = "/data/raw/credit_card_data.csv"  # Adjust this path as needed
#         #df = pd.read_csv(raw_path)
#         #print(f"Data loaded from {raw_path}")
#     processor = DataProcessor(df)
#     monitor = NullValueMonitor()

#     monitor.log_null_stats(df, processor=processor)