try:
    from src.dataloaders.load_data import DataLoader
    USE_DATALOADER = True
except (ImportError, FileNotFoundError) as e:
    print(f"Warning: Could not import DataLoader: {e}")
    USE_DATALOADER = False



from src.dataloaders.load_data import DataLoader, CategoricalColumns, TargetColumn

def load_and_explore_data():
    """
    Load data using the existing DataLoader class and perform basic exploration.
    """
    print("Loading data...")
    
    # Initialize the DataLoader
    data_loader = DataLoader()
    
    # Load raw data
    raw_data = data_loader.load_data_raw()
    
    print(f"Raw data loaded successfully. Shape: {raw_data.shape}")
    target_col = TargetColumn.get_column()
    
    raw_data[target_col] = raw_data[target_col].map({0: "No Fraud", 1: "Fraud"})
    
    outputpath  = "/data//LLM_data_prepration.csv"
    raw_data.to_csv(outputpath, index=False)
    
    
    return raw_data

if __name__ == "__main__":
    data = load_and_explore_data()
    
    # Display a sample of the data
    print("\n--- Data Sample ---")
    print(data.head())
    
    # Save a copy of the data if needed
    # data.to_csv("data_copy.csv", index=False)
    
    print("\nData loading and exploration complete.")
    
