import os
import pandas as pd
import pathlib
from pathlib import Path
from src.features.featureengineering import FeatureEngineer
from src.dataloaders.load_data import DataLoader, CategoricalColumns, TargetColumn
from src.utils.confighandler import ConfigReader
from src.features.mlflowsetup import MLflowConfig
from src.features.nullvaluehandler import DataProcessor, NullValueMonitor
from src.features.normalization import Noramalization
from src.features.handlingimbalance import ImbalanceMonitor
from src.models.train_model import ModelTrainer, load_data, save_model
from src.models.evaluate import ModelEvaluator  # Import the ModelEvaluator class
from sklearn.model_selection import train_test_split
import joblib

# ---------------------- Configuration ---------------------- #
print('----------------------Abhinavkaushikl git push---------------------------------------------')
categorical_cols = CategoricalColumns.get_all_columns()
target_col = TargetColumn.get_column()

# ---------------------- Pipeline ---------------------- #
def dataprocessing_pipeline():
    try:
        print("Attempting to load raw data using DataLoader...")
        data_loader = DataLoader()
        df = data_loader.load_data_raw()
        
        data_loader = DataLoader()
        raw_output_path = data_loader.datadumper()
        
        print("Raw data loaded successfully.")
    except Exception as e:
        print(f"Error loading data with DataLoader: {e}")
        raise

    # Null value handling
    processor = DataProcessor(df)
    null_monitor = NullValueMonitor()
    null_monitor.log_null_stats(df, processor=processor)

    # Feature Engineering
    print("Running feature engineering...")
    fe = FeatureEngineer()
    df_fe = fe.transform(df)
    fe.save(df_fe, raw_output_path)
    print(f"Feature engineering complete. Saved to {raw_output_path}")

    # Reload processed data (clean separation)
    df_processed = data_loader.load_data_processed()

    # Normalization
    print("Normalizing data...")
    normalizer = Noramalization(
        categorical_cols=categorical_cols,
        target_col=target_col,
        save_path=raw_output_path,
        save_format='csv'
    )
    df_normalized = normalizer.preprocess(df_processed)
    normalizer.save(df_normalized)
    print(f"Normalization complete. Saved to {raw_output_path}")

    # Imbalance Handling
    print("Checking and handling class imbalance...")
    imbalance_monitor = ImbalanceMonitor()
    X_resampled, y_sampled = imbalance_monitor.apply_smote_if_needed(df_normalized, target_col=target_col)
    print("Class balancing complete.")
    
    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42
    )
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Model Training
    print("Training model...")
    model_save_path = "models/randomforestor.pkl"
    
    # Create ModelTrainer with the correct parameters
    model_trainer = ModelTrainer(X_train, y_train, model_type="rf")
    model = model_trainer.train()
    
    # Save the model explicitly
    save_model(model, model_save_path)
    print("Model training complete.")
    
    # Model Evaluation
    print("Evaluating model...")
    evaluator = ModelEvaluator(output_dir="evaluation_results")
    
    # Evaluate using the trained model
    evaluator.evaluate_model(model, X_test, y_test, model_name="RandomForest")
    print("Model evaluation complete.")
    
    return X_resampled, y_sampled

# ---------------------- Model Training ---------------------- #

if __name__ == "__main__":
    pf = dataprocessing_pipeline()
    print("Data processing pipeline completed successfully!")
    print("Pipeline completed successfully!")
