import subprocess
import sys

def install_mlflow():
    print("Installing langchain...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
        print("MLflow installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing langchain: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_mlflow()