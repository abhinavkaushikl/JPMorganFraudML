import yaml
from pathlib import Path
import logging
import configparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigReader:
    def __init__(self, config_path: Path = None):
        if config_path is None:
            # Adjust this path based on your project structure:
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        self.config_path = config_path
        
        print(f"[ConfigReader] Loading config from: {self.config_path}")
        self.config = self.read_config()

    def read_config(self):
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
                if not config:
                    raise ValueError("Configuration file is empty or invalid.")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading config: {e}")
            raise

    def get_section(self, section: str):
        if self.config is None:
            raise ValueError("Configuration not loaded.")
        return self.config.get(section, {})
