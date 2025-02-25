import tensorflow as tf
import numpy as np
from .simclr import simclr_loss

def train_simclr(features, model, optimizer, epochs=10, batch_size=32):
    """Trains SimCLR with contrastive learning on augmented pose data."""
    for epoch in range(epochs):
        indices = np.arange(features.shape[0])
        np.random.shuffle(indices)
        for i in range(0, features.shape[0], batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = features[batch_indices]
            augmented_i = time_series_augmentation(batch)
            augmented_j = time_series_augmentation(batch)
            with tf.GradientTape() as tape:
                z_i = model(augmented_i)
                z_j = model(augmented_j)
                loss = simclr_loss(z_i, z_j)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")
