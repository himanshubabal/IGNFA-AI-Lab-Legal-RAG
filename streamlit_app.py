"""
Streamlit Community Cloud entry point for RAG-Anything.

This file serves as the entry point for Streamlit Community Cloud deployment.

To run on Streamlit Community Cloud:
1. Push this repository to GitHub
2. Connect to Streamlit Community Cloud  
3. Set the main file path to: streamlit_app.py
4. Streamlit will automatically detect and run this file

Alternatively, you can point directly to:
   raganything/webui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the Streamlit app
# Importing the module will execute all Streamlit components
from raganything.webui import streamlit_app

# Explicitly call main() to ensure the app runs
# Streamlit will execute this file and all the Streamlit commands
# defined in the imported module will be available
streamlit_app.main()

