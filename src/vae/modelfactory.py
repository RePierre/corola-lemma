import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os.path as path
import datetime


class Sampling(layers.Layer):
    """
    A layer that uses (z_mean, z_log_var) to sample z,
    the vector that encodes the lemma.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_model(intermediate_dim, latent_dim, original_dim,
                    reconstruct_lemma):
    """
    Builds and compiles the Variational AutoEncoder model.

    Parameters
    ----------
    intermediate_dim: integer
        The number of dimensions of th output space for the intermediate layer
        of the inference network.
    latent_dim: integer
        The number of dimensions of the latent space.
    original_dim: integer
        The number of dimensions of the word/lemma embeddings.
    reconstruct_lemma: boolean
        Specifies whether to reconstruct lemma or word.

    Returns
    -------
    vae: keras.Model
        Variational AutoEncoder model.
    """
    encoder_inputs = keras.Input(shape=(original_dim, ))
    num_dims_1 = max(intermediate_dim, 2 * int(original_dim / 3))
    num_dims_2 = min(intermediate_dim, 2 * int(original_dim / 3))
    encoder = _build_encoder(num_dims_1, encoder_inputs, num_dims_2,
                             latent_dim)
    encoder.summary()
    decoder = _build_decoder(latent_dim, num_dims_2, num_dims_1, original_dim)
    vae = VAE(encoder,
              decoder,
              original_dim,
              reconstruct_lemma=reconstruct_lemma)
    vae.compile(optimizer=keras.optimizers.Adam())
    return vae, encoder, decoder


def _build_decoder(latent_dim, num_dims_2, num_dims_1, original_dim):
    latent_inputs = keras.Input(shape=(latent_dim, ))
    x = layers.Dense(units=num_dims_2, activation='relu')(latent_inputs)
    x = layers.Dense(units=num_dims_1, activation='relu')(x)
    decoder_outputs = layers.Dense(units=original_dim, activation='relu')(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder


def _build_encoder(num_dims_1, encoder_inputs, num_dims_2, latent_dim):
    x = layers.Dense(units=num_dims_1, activation='relu')(encoder_inputs)
    x = layers.Dense(units=num_dims_2, activation='relu')(x)
    z_mean = layers.Dense(units=latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(units=latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z],
                          name="encoder")
    return encoder


class VAE(keras.Model):
    """
    Variational AutoEncoder model.
    """
    def __init__(self,
                 encoder,
                 decoder,
                 original_dim,
                 reconstruct_lemma=True,
                 **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.original_dim = original_dim
        self.reconstruct_lemma = reconstruct_lemma

    def train_step(self, data):
        word, lemma = data
        target = lemma if self.reconstruct_lemma else word
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(word)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(target, reconstruction))
            reconstruction_loss *= self.original_dim
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def build_model_callbacks(tensorboard_log_dir='logs',
                          use_early_stopping=True,
                          early_stopping_patience=2):
    """
    Builds the list of callbacks for model training.

    Parameters
    ----------
    tensorboard_log_dir:string, optional
        The root path where to save TensorBoard logs.
        Default value is 'logs' dir in current directory.
    use_early_stopping: boolean, optional
        Specifies whether to add or not EarlyStopping callback to the model.
        Default is True.
    early_stopping_patience: integer, optional
        Specifies how many epochs to wait before early stopping.
        Default is 2.

    Returns
    -------
    callbacks: list of callbacks
        The callbacks to use for model training.
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    log_dir = path.join(tensorboard_log_dir, current_time)
    tensorboard_display = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    callbacks = [tensorboard_display]
    if use_early_stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=2,
                                                          monitor='loss')
        callbacks.append(early_stopping)

    return callbacks
