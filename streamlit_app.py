"""
Streamlit Community Cloud entry point for RAG-Anything.

This file serves as the entry point for Streamlit Community Cloud deployment.

To run on Streamlit Community Cloud:
1. Push this repository to GitHub
2. Connect to Streamlit Community Cloud  
3. Set the main file to: streamlit_app.py
4. Streamlit will automatically detect and run this file

Alternatively, you can point directly to:
   raganything/webui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the Streamlit app module
# This makes all the Streamlit components available for execution
from raganything.webui import streamlit_app

# Run the main function when executed
if __name__ == "__main__":
    streamlit_app.main()

