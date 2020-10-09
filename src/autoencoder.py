from keras.layers import Input, Dense
from keras import Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import os.path as path


# Create base class in order to provide proper callbacks to the model
def build_autoencoder(verbose=0):
    """
    Builds and autoencoder model that can be used by scikit_learn API.

    Parameters
    ----------
    verbose: 0, 1 or 2
        The verbosity level of the model.

    Returns
    -------
    keras.wrappers.scikit_learn.KerasRegressor
        The wrapper around the simple autoencoder.
    """
    return KerasRegressor(build_fn=_build_autoencoder, verbose=verbose)


def _build_autoencoder(sample_dim=300,
                       encoding_dim=16,
                       encoder_activation='relu',
                       decoder_activation='sigmoid',
                       optimizer='adam',
                       loss='mean_squared_error'):
    """
    Builds a simple autoencoder model.

    Parameters
    ----------
    sample_dim: int
        The number of dimensions of a training sample.
    encoding_dim: int
        The number of dimensions for the encoded data.
    encoder_activation: string
        The name of the activation function for the encoder.
    decoder_activation: string
        The name of the activation function for the decoder.
    optimizer: string
        The name of the optimizer used by the model.
    loss: string
        The name of the loss function.

    Returns
    -------
    keras.Model
        The compiled model.
    """
    input_layer = Input(shape=(sample_dim, ))
    encoded = Dense(encoding_dim, activation=encoder_activation)(input_layer)
    decoded = Dense(sample_dim, activation=decoder_activation)(encoded)
    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder


def _build_model_callbacks(sample_dim, encoding_dim, encoder_activation,
                           decoder_activation, optimizer, loss,
                           early_stopping_patience, reduce_lr_patience,
                           reduce_lr_factor):
    """
    Builds the list of callbacks for model training based on model parameters.

    Parameters
    ----------
    sample_dim: int
        The number of dimensions of a training sample.
    encoding_dim: int
        The number of dimensions for the encoded data.
    encoder_activation: string
        The name of the activation function for the encoder.
    decoder_activation: string
        The name of the activation function for the decoder.
    optimizer: string
        The name of the optimizer used by the model.
    loss: string
        The name of the loss function.
    early_stopping_patience: int
        The number of epochs to wait before early stopping.
    reduce_lr_patience: int
        The number of epochs to wait before reducing learning rate.
    reduce_lr_factor: float
        The factor by which to reduce learning rate.

    Returns
    -------
    list of kears.callbacks.Callback
        The list of callbacks to be used in model training.
    """
    log_dir = path.join(
        './logs', "sd-{}--ed-{}--ea-{}--da-{}--opt-{}--loss-{}".format(
            sample_dim, encoding_dim, encoder_activation, decoder_activation,
            optimizer, loss))
    tensorboard = TensorBoard(log_dir=log_dir,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True,
                              write_grads=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    reduce_lr = ReduceLROnPlateau(factor=reduce_lr_factor,
                                  patience=reduce_lr_patience)
    return [tensorboard, early_stopping, reduce_lr]
