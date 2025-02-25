import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def simclr_loss(z_i, z_j, temperature=0.1):
    """Computes NT-Xent loss for contrastive learning."""
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)
    z = tf.math.l2_normalize(z, axis=1)
    similarity_matrix = tf.matmul(z, z, transpose_b=True) / temperature

    labels = tf.concat([tf.range(batch_size), tf.range(batch_size)], axis=0)
    labels = tf.cast(labels, tf.int64)
    mask = tf.eye(2 * batch_size, dtype=tf.bool)
    logits = tf.boolean_mask(similarity_matrix, ~mask)
    logits = tf.reshape(logits, [2 * batch_size, -1])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

class SimCLR(tf.keras.Model):
    """SimCLR model for self-supervised learning on time-series pose data."""
    def __init__(self, encoder, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def call(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

encoder = tf.keras.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(None, 42)),  # 14 body parts × 3 dimensions
    layers.MaxPooling1D(),
    layers.Conv1D(128, 3, activation='relu'),
    layers.LSTM(128, return_sequences=False),
    layers.Dense(256, activation='relu')
])

projection_head = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(128)
])

simclr_model = SimCLR(encoder, projection_head)
optimizer = tf.keras.optimizers.Adam()
