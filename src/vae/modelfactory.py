from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
import tensorflow as tf
import os.path as path
import datetime


def build_vae_model(intermediate_dim, latent_dim, original_dim):

    x = Input(shape=(original_dim, ), name="x")
    # Nonlinearity that maps z ~ N(0,1) to the output space?
    # (From Doersch et al.)
    h = Dense(intermediate_dim, activation='relu', name="h")(x)
    z_mu = Dense(latent_dim, name="z_mu")(h)
    z_log_var = Dense(latent_dim, name="z_log_var")(h)

    z_mu, z_log_var = KLDivergenceLayer(name="KLDivergence")([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(0.5 * t), name="z_sigma")(z_log_var)
    eps = Input(tensor=K.random_normal((K.shape(x)[0], latent_dim),
                                       mean=0.0,
                                       stddev=1.0),
                name="eps")
    z_eps = Multiply(name="z_eps")([z_sigma, eps])
    z = Add(name="z")([z_mu, z_eps])

    # Used to compute p(x_pred|z)
    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])
    x_pred = decoder(z)
    vae = Model(inputs=[x, eps], outputs=x_pred)
    vae.compile(optimizer='rmsprop', loss=nll)
    encoder = Model(x, z_mu)
    return vae, encoder, decoder


class KLDivergenceLayer(Layer):
    """
    Layer that adds KL Divergence to the final model loss.

    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var),
                                axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs


def nll(y_true, y_pred):
    """
    Bernoulli negative log likelihood.
    """
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


def build_model_callbacks():
    """
    Builds the list of callbacks for model training.

    Returns
    -------
    callbacks: list of callbacks
        The callbacks to use for model training.
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    log_dir = path.join('./logs/', current_time)
    tensorboard_display = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=2,
                                                      monitor='loss')
    return [early_stopping, tensorboard_display]
