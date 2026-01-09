#!/usr/bin/env python3
"""Quick start script for bankruptcy prediction model."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🏦 Bankruptcy Prediction Model - Quick Start")
    print("=" * 60)
    print("⚠️  DISCLAIMER: This is for research purposes only!")
    print("   Do not use for actual investment decisions.")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies. Please check requirements.txt")
        sys.exit(1)
    
    # Run training
    if not run_command("python scripts/train.py --n-samples 500 --output-dir assets", "Training models"):
        print("❌ Training failed. Please check the error messages above.")
        sys.exit(1)
    
    # Check if models were created
    assets_dir = Path("assets")
    if not assets_dir.exists():
        print("❌ Assets directory not created!")
        sys.exit(1)
    
    model_files = list(assets_dir.glob("*_model.pkl"))
    if not model_files:
        print("❌ No model files found!")
        sys.exit(1)
    
    print(f"✅ Found {len(model_files)} trained models")
    
    # Launch demo
    print(f"\n{'='*60}")
    print("🚀 Ready to launch Streamlit demo!")
    print("=" * 60)
    print("Run the following command to start the interactive demo:")
    print("streamlit run demo/app.py")
    print("\nOr run the Jupyter notebook:")
    print("jupyter notebook notebooks/demo.ipynb")
    
    # Ask if user wants to launch demo
    try:
        launch_demo = input("\nWould you like to launch the Streamlit demo now? (y/n): ").lower().strip()
        if launch_demo in ['y', 'yes']:
            print("\n🚀 Launching Streamlit demo...")
            print("The demo will open in your browser automatically.")
            print("Press Ctrl+C to stop the demo.")
            subprocess.run("streamlit run demo/app.py", shell=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error launching demo: {e}")
        print("You can manually run: streamlit run demo/app.py")


if __name__ == "__main__":
    main()
