#!/usr/bin/env python3
"""
Quick setup and run script for the Renewable Energy Prediction Streamlit App
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scikit-learn', 'tensorflow', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'Database.csv',
        'train_multi_output.csv', 
        'test_multi_output.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def main():
    print("🚀 Renewable Energy Prediction App Setup")
    print("=" * 50)
    
    # Check data files
    print("📁 Checking data files...")
    missing_files = check_data_files()
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all data files are in the current directory.")
        return False
    else:
        print("✅ All data files found!")
    
    # Check requirements
    print("\n📦 Checking Python packages...")
    missing_packages = check_requirements()
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        if not install_requirements():
            return False
    else:
        print("✅ All required packages are installed!")
    
    # Run the app
    print("\n🌐 Starting Streamlit application...")
    print("The app will open in your default browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app_complete.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
