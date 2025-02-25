"""Synthetic data generation for Stella pipeline."""
import numpy as np
import scipy.interpolate

def generate_synthetic_data(n_samples=1000, n_timesteps=120, n_features=14*3):
    """Generates structured synthetic pose data."""
    np.random.seed(42)
    data = np.random.randn(n_samples, n_timesteps, n_features)
    return data

def handle_missing_values(data):
    """Interpolates missing values using cubic splines."""
    for i in range(data.shape[0]):  # Iterate over samples
        for j in range(data.shape[2]):  # Iterate over features
            nan_mask = np.isnan(data[i, :, j])
            if np.any(nan_mask):
                x_valid = np.where(~nan_mask)[0]
                y_valid = data[i, x_valid, j]
                f_interp = scipy.interpolate.CubicSpline(x_valid, y_valid, extrapolate=True)
                data[i, nan_mask, j] = f_interp(np.where(nan_mask)[0])
    return data
