from sklearn.model_selection import train_test_split
import numpy as np


def reshape(array, sample_dim):
    """
    Reshapes the collection of arrays into a single numpy array.

    Parameters
    ----------
    array: list of numpy arrays
        The list of numpy arrays to reshape into a single array.
    sample_dim: int
        The number of dimensions of individual samples.

    Returns
    -------
    numpy.ndarray
        The single array obtained from reshaping.
    """
    return np.reshape(array, (len(array), sample_dim))


def split_and_reshape(ds):
    """
    Splits the data into train/test sets and reshapes into single numpy arrays.

    Parameters
    ----------
    ds: input.WordEmebeddingsDataset
        The dataset containing training and test data.
    """
    we_train, we_test, le_train, le_test = train_test_split(
        ds.word_embeddings, ds.lemma_embeddings, random_state=2020)
    sample_dim = we_train[0].shape[0]
    print('Sample dimensions: {}'.format(sample_dim))
    return reshape(we_train,
                   sample_dim), reshape(we_test, sample_dim), reshape(
                       le_train, sample_dim), reshape(le_test, sample_dim)
