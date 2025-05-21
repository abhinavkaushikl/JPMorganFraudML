import pandas as pd
import os
from enum import Enum
try:
    from src.utils.confighandler import ConfigReader
except ImportError as e:
    raise ImportError(f"Failed to import ConfigReader. Ensure 'confighandler.py' is available. Details: {e}")

class DataLoader:
    """
    Loads data based on file paths defined in config.yaml.
    """
    def __init__(self):
        # Initialize the ConfigReader to read the main config file
        self.config_reader = ConfigReader()
        self.data_paths = {}
        self._load_config()

    def _load_config(self):
        """Load file paths from config under the 'data' section."""
        try:
            # Get the 'data' section directly from the config
            self.data_paths = self.config_reader.get_section("data")
            if not self.data_paths:
                raise ValueError("The 'data' section in config.yaml is missing or empty.")
        except Exception as e:
            raise RuntimeError(f"Error loading data config: {e}")

    def load_csv(self, key: str) -> pd.DataFrame:
        """
        Load a CSV file based on a key from the config file's 'data' section.
        Parameters:
            key (str): Key defined in the config (e.g., 'raw_data').
        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        try:
            file_path = self.data_paths.get(key)
            if not file_path:
                raise KeyError(f"No path found for key '{key}' in the 'data' section.")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
            df = pd.read_csv(file_path)
            print(f"Loaded data from: {file_path} (rows: {df.shape[0]}, cols: {df.shape[1]})")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data for key '{key}': {e}")
    
    # datadump
    def load_csv(self, key: str) -> pd.DataFrame:
        """
        Load a CSV file based on a key from the config file's 'data' section.
        Parameters:
            key (str): Key defined in the config (e.g., 'raw_data').
        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        try:
            file_path = self.data_paths.get(key)
            if not file_path:
                raise KeyError(f"No path found for key '{key}' in the 'data' section.")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
            df = pd.read_csv(file_path)
            print(f"Loaded data from: {file_path} (rows: {df.shape[0]}, cols: {df.shape[1]})")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data for key '{key}': {e}")
        
        
    
    def load_data_raw(self) -> pd.DataFrame:
        """
        Load the main raw data file.
        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        return self.load_csv("raw_data_path")
    
    def load_data_processed(self) -> pd.DataFrame:
        """
        Load the main raw data file.
        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        return self.load_csv("processed_data_path")
    
    
    
    
    def save_csv(self,key: str) -> str:
        try:
            file_path = self.data_paths.get(key)
        except Exception as e:
           raise RuntimeError(f"Failed to save data for key '{key}': {e}")
        return file_path
       
    def datadumper(self) -> str:
        """
        Load the main raw data file.
        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        return self.save_csv("processed_data_path")
    
    

# Define enums for categorical columns and target column
class CategoricalColumns(Enum):
    """Enum representing categorical columns in the dataset."""
    GENDER = 'Gender'
    STATE = 'State'
    CITY = 'City'
    ACCOUNT_TYPE = 'Account_Type'
    TRANSACTION_TYPE = 'Transaction_Type'
    MERCHANT_CATEGORY = 'Merchant_Category'
    TRANSACTION_DEVICE = 'Transaction_Device'
    TRANSACTION_LOCATION = 'Transaction_Location'
    DEVICE_TYPE = 'Device_Type'
    TRANSACTION_CURRENCY = 'Transaction_Currency'
    TRANSACTION_DESCRIPTION = 'Transaction_Description'
    AGE_GROUP = 'Age_Group'
    USER_HOME_CITY = 'User_Home_City'
    CUSTOMER_ID = 'Customer_ID'
    TRANSACTION_DATE = 'Transaction_Date'
    TRANSACTION_TIME = 'Transaction_Time'
    MERCHANT_ID = 'Merchant_ID'
    
    @classmethod
    def get_all_columns(cls) -> list:
        """Return a list of all categorical column names."""
        return [col.value for col in cls]

class TargetColumn(Enum):
    """Enum representing the target column in the dataset."""
    IS_FRAUD = 'Is_Fraud'
    
    @classmethod
    def get_column(cls) -> str:
        """Return the target column name."""
        return cls.IS_FRAUD.value
