#!/usr/bin/env python3
"""
Hugging Face Spaces entry point
This file ensures the app starts correctly in the Spaces environment
"""

import subprocess
import sys
import os

def main():
    """Main entry point for Hugging Face Spaces"""
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Run the Streamlit app
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit app: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
