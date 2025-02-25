"""Feature extraction for Stella pipeline."""
import numpy as np

def compute_velocity(data):
    """Computes velocity from positional data."""
    return np.diff(data, axis=1)

def compute_acceleration(velocity):
    """Computes acceleration from velocity."""
    return np.diff(velocity, axis=1)

def compute_jerk(acceleration):
    """Computes jerk from acceleration."""
    return np.diff(acceleration, axis=1)
