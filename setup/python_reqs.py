import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = [
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn"
]

for package in packages:
    try:
        print(f"Installing {package}...")
        install(package)
        print(f"{package} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}.")
