try:
    from src.utils.confighandler import ConfigReader
except ImportError as e:
    print("Import Error. Check module path", e) 
try:
    import mlflow
except ImportError:
    print("MLflow package is not installed. Please install using: pip install mlflow")

class MLflowConfig:
    def __init__(self):
        # Initialize config reader and read the 'mlflow' section
        config_reader = ConfigReader()
        mlflow_config = config_reader.get_section("mlflow")

        # Extract MLflow parameters
        self.tracking_uri = mlflow_config.get("tracking_uri")
        self.experiment_name = mlflow_config.get("experiment_name")

        # Validate configuration
        if not self.tracking_uri or not self.experiment_name:
            raise ValueError("Both 'tracking_uri' and 'experiment_name' must be set in config.yaml")

    def setup(self):
        """Configure MLflow with tracking URI and experiment name."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
