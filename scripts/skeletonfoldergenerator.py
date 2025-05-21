import os

project_name = "bank_model_project"

folders = [
    "config",
    "data/raw",
    "data/processed",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/utils",
    "experiments",
    "mlruns",
    "models",
    "scripts",
    "tests"
]

files = {
    "README.md": "",
    "requirements.txt": "",
    ".gitignore": "*.pyc\n__pycache__/\n.env\nmlruns/\nmodels/\n",
    ".env": "# Store secrets like API keys or environment variables",
    "config/config.yaml": "# Training and model parameters",
    "config/logging.yaml": "# Logging config",
    "notebooks/eda.ipynb": "# Jupyter notebook for EDA",
    "src/__init__.py": "",
    "src/data/__init__.py": "",
    "src/data/load_data.py": "# Load data script",
    "src/data/preprocess.py": "# Preprocessing pipeline",
    "src/features/feature_selector.py": "# Feature engineering logic",
    "src/models/train_model.py": "# Training script",
    "src/models/evaluate.py": "# Evaluation metrics",
    "src/models/predict.py": "# Inference logic",
    "src/models/register_model.py": "# MLflow model registry handler",
    "src/utils/logger.py": "# Custom logger",
    "experiments/baseline_run.py": "# Script to run baseline experiment",
    "scripts/run_train.sh": "#!/bin/bash\npython src/models/train_model.py",
    "scripts/run_predict.sh": "#!/bin/bash\npython src/models/predict.py",
    "tests/test_training.py": "# Unit test for training pipeline",
    "setup.py": "# For packaging this project as a module"
}

def create_structure():
    print(f"Creating project structure for: {project_name}")
    os.makedirs(project_name, exist_ok=True)
    os.chdir(project_name)

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    for file, content in files.items():
        with open(file, "w") as f:
            f.write(content)
        print(f"Created file: {file}")

    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_structure()
