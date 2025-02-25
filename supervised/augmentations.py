from scipy import signal
import numpy as np

def time_series_augmentation(batch):
    """Applies augmentations like time warping, jittering, and scaling."""
    augmented_batch = []
    for series in batch:
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            series = signal.resample(series, int(series.shape[0] * factor))
            series = np.pad(series, ((0, max(0, batch.shape[1] - series.shape[0])), (0, 0)), 'edge')
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.05, series.shape)
            series = series + noise
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            series = series * scale
        augmented_batch.append(series)
    return np.array(augmented_batch, dtype=np.float32)
