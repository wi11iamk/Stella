"""Data loader for Stella pipeline."""
import numpy as np
import pandas as pd

def load_pose_data(file_path):
    """Loads pose data from a CSV file."""
    data = pd.read_csv(file_path)
    return data.values  # Returns as NumPy array
