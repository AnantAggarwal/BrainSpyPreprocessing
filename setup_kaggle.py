#!/usr/bin/env python3
"""
Setup script for Kaggle notebooks to run BrainSpy preprocessing
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    packages = [
        'tqdm',
        'nibabel',
        'numpy'
    ]
    
    print("Installing Python dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def setup_fsl():
    """Setup FSL environment for Kaggle"""
    print("Setting up FSL environment...")
    
    # Common FSL paths in Kaggle
    fsl_paths = [
        "/root/fsl",
        "/usr/local/fsl",
        "/usr/share/fsl"
    ]
    
    fsl_found = False
    for fsl_path in fsl_paths:
        if os.path.exists(fsl_path):
            os.environ['FSLDIR'] = fsl_path
            os.environ['PATH'] = f"{fsl_path}/bin:{os.environ.get('PATH', '')}"
            print(f"✓ FSL found at: {fsl_path}")
            fsl_found = True
            break
    
    if not fsl_found:
        print("⚠ FSL not found. You may need to install it manually.")
        print("Try running: !apt-get update && apt-get install -y fsl")

def setup_robex():
    """Setup ROBEX for brain extraction"""
    print("Setting up ROBEX...")
    
    robex_dir = Path("ROBEX")
    if robex_dir.exists():
        # Make ROBEX script executable
        robex_script = robex_dir / "runROBEX.sh"
        if robex_script.exists():
            robex_script.chmod(0o755)
            print("✓ ROBEX script made executable")
        else:
            print("⚠ ROBEX script not found")
    else:
        print("⚠ ROBEX directory not found")

def create_example_usage():
    """Create example usage script"""
    example_script = '''# Example usage in Kaggle notebook

# 1. Clone the repository
!git clone https://github.com/your-username/BrainSpyPreprocessing.git
%cd BrainSpyPreprocessing

# 2. Run setup
!python setup_kaggle.py

# 3. Run preprocessing (example)
!python preprocess.py \\
    --base_dir /kaggle/input/your-dataset \\
    --robex \\
    --mni_reg \\
    --segmentation \\
    --n_jobs 4

# Or run individual steps:
# ROBEX only
!python preprocess.py --base_dir /kaggle/input/your-dataset --robex

# MNI registration only  
!python preprocess.py --base_dir /kaggle/input/your-dataset --mni_reg

# Segmentation only
!python preprocess.py --base_dir /kaggle/input/your-dataset --segmentation
'''
    
    with open("kaggle_usage_example.py", "w") as f:
        f.write(example_script)
    print("✓ Created kaggle_usage_example.py")

def main():
    """Main setup function"""
    print("Setting up BrainSpy preprocessing for Kaggle notebooks...")
    print("=" * 50)
    
    install_dependencies()
    print()
    
    setup_fsl()
    print()
    
    setup_robex()
    print()
    
    create_example_usage()
    print()
    
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Make sure your dataset is in the expected ADNI folder structure")
    print("2. Run the preprocessing script with appropriate arguments")
    print("3. Check the example usage file for more details")

if __name__ == "__main__":
    main() 