import subprocess
import sys

def install_mlflow():
    print("Installing MLflow...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        print("MLflow installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing MLflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_mlflow()