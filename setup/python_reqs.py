import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn",
    "h5py",
    "torch"
]

for package in packages:
    try:
        print(f"Installing {package}...")
        install(package)
        print(f"{package} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}.")
