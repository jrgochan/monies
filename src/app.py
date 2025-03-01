"""
Main entry module for the Monies application.
This module imports and reexports the main function from the root app.py.
"""
import os
import sys
from importlib import import_module

# Add the root directory to the Python path to import from root app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use import_module to avoid circular imports
app_module = import_module("app")
main = app_module.main
