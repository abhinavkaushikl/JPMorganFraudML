import os
import pandas as pd
import pathlib
from pathlib import Path
import multiprocessing
import joblib
from sklearn.model_selection import train_test_split

# Local imports
from src.features.featureengineering import FeatureEngineer
from src.dataloaders.load_data import DataLoader, CategoricalColumns, TargetColumn
from src.utils.confighandler import ConfigReader
from src.features.mlflowsetup import MLflowConfig
from src.features.nullvaluehandler import DataProcessor, NullValueMonitor
from src.features.normalization import Noramalization
from src.features.handlingimbalance import ImbalanceMonitor
from src.models.train_model import ModelTrainer, load_data, save_model
from src.models.evaluate import ModelEvaluator


def train_model_process(model_type, X_train, y_train, model_save_path):
    """
    Train a specific model in a separate process.
    
    Args:
        model_type (str): Type of model to train ('rf', 'lgbm', etc.)
        X_train (DataFrame): Training features
        y_train (Series): Training target
        model_save_path (str): Path to save the trained model
        
    Returns:
        str: Path where the model was saved
    """
    print(f"Training {model_type} model...")
    model_trainer = ModelTrainer(X_train, y_train, model_type=model_type)
    model = model_trainer.train()
    save_model(model, model_save_path)
    print(f"{model_type} model training complete.")
    return model_save_path


def evaluate_model_process(model_path, X_test, y_test, model_name):
    """
    Evaluate a specific model in a separate process.
    
    Args:
        model_path (str): Path to the saved model
        X_test (DataFrame): Test features
        y_test (Series): Test target
        model_name (str): Name of the model for reporting
    """
    print(f"Evaluating {model_name} model...")
    model = joblib.load(model_path)
    evaluator = ModelEvaluator(output_dir="evaluation_results")
    evaluator.evaluate_model(model, X_test, y_test, model_name=model_name)
    print(f"{model_name} model evaluation complete.")


def load_raw_data():
    """Load raw data and return dataframe and output path."""
    try:
        print("Loading raw data...")
        data_loader = DataLoader()
        df = data_loader.load_data_raw()
        raw_output_path = data_loader.datadumper()
        print("Raw data loaded successfully.")
        return df, raw_output_path
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def handle_null_values(df):
    """Process and monitor null values in the dataframe."""
    print("Handling null values...")
    processor = DataProcessor(df)
    null_monitor = NullValueMonitor()
    null_monitor.log_null_stats(df, processor=processor)
    return df


def perform_feature_engineering(df, output_path):
    """Apply feature engineering transformations."""
    print("Running feature engineering...")
    fe = FeatureEngineer()
    df_fe = fe.transform(df)
    fe.save(df_fe, output_path)
    print(f"Feature engineering complete. Saved to {output_path}")
    return df_fe


def normalize_data(df_processed, output_path):
    """Normalize and encode the processed data."""
    print("Normalizing data...")
    categorical_cols = CategoricalColumns.get_all_columns()
    target_col = TargetColumn.get_column()
    
    normalizer = Noramalization(
        categorical_cols=categorical_cols,
        target_col=target_col,
        save_path=output_path,
        save_format='csv'
    )
    df_normalized = normalizer.preprocess(df_processed)
    normalizer.save(df_normalized)
    print(f"Normalization complete. Saved to {output_path}")
    return df_normalized


def handle_class_imbalance(df_normalized):
    """Check and handle class imbalance using SMOTE if needed."""
    print("Checking and handling class imbalance...")
    target_col = TargetColumn.get_column()
    imbalance_monitor = ImbalanceMonitor()
    X_resampled, y_sampled = imbalance_monitor.apply_smote_if_needed(df_normalized, target_col=target_col)
    print("Class balancing complete.")
    return X_resampled, y_sampled


def split_data(X, y):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def train_models_parallel(X_train, y_train, model_configs):
    """Train multiple models in parallel using multiprocessing."""
    print("Training models in parallel...")
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create a pool of processes for training
    with multiprocessing.Pool(processes=len(model_configs)) as pool:
        # Start training processes
        results = [
            pool.apply_async(
                train_model_process,
                args=(config["model_type"], X_train, y_train, config["save_path"])
            )
            for config in model_configs
        ]
        
        # Get results (wait for all processes to complete)
        model_paths = [result.get() for result in results]
    
    print("All models trained successfully.")
    return model_paths


def evaluate_models_parallel(model_configs, X_test, y_test):
    """Evaluate multiple models in parallel using multiprocessing."""
    print("Evaluating models in parallel...")
    
    # Create directory for evaluation results if it doesn't exist
    os.makedirs("evaluation_results", exist_ok=True)
    
    with multiprocessing.Pool(processes=len(model_configs)) as pool:
        # Start evaluation processes
        eval_results = [
            pool.apply_async(
                evaluate_model_process,
                args=(config["save_path"], X_test, y_test, config["name"])
            )
            for config in model_configs
        ]
        
        # Wait for all evaluation processes to complete
        [result.get() for result in eval_results]
    
    print("All models evaluated successfully.")


def dataprocessing_pipeline():
    """
    Execute the complete data processing and model training pipeline.
    
    Returns:
        list: Paths to the trained models
    """
    # Load raw data
    df, raw_output_path = load_raw_data()
    
    # Handle null values
    df = handle_null_values(df)
    
    # Feature engineering
    df_fe = perform_feature_engineering(df, raw_output_path)
    
    # Reload processed data for clean separation
    data_loader = DataLoader()
    df_processed = data_loader.load_data_processed()
    
    # Normalize data
    df_normalized = normalize_data(df_processed, raw_output_path)
    
    # Handle class imbalance
    X_resampled, y_sampled = handle_class_imbalance(df_normalized)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_sampled)
    
    # Define model configurations
    model_configs = [
        {"model_type": "rf", "save_path": "models/randomforest.pkl", "name": "RandomForest"},
        {"model_type": "lgbm", "save_path": "models/lightgbm.pkl", "name": "LightGBM"}
    ]
    
    # Train models in parallel
    model_paths = train_models_parallel(X_train, y_train, model_configs)
    
    # Evaluate models in parallel
    evaluate_models_parallel(model_configs, X_test, y_test)
    
    return model_paths


if __name__ == "__main__":
    print("Starting ML pipeline...")
    
    # Set the start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    try:
        model_paths = dataprocessing_pipeline()
        print("Data processing pipeline completed successfully!")
        print(f"Trained models saved at: {', '.join(model_paths)}")
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
