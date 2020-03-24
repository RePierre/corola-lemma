from argparse import ArgumentParser
from input import WordEmebeddingsDataset
from modelfactory import build_autoencoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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


def run(args):
    ds = WordEmebeddingsDataset(
        word_embeddings_file=args.word_embeddings_file,
        lemma_embeddings_file=args.lemma_embeddings_file,
        word_lemma_pairs_file=args.word_lemma_pairs_file)
    ds.initialize()
    if args.save_common_wl_pairs_to:
        ds.save_word_lemma_pairs(args.save_common_wl_pairs_to)

    model = build_autoencoder()
    param_grid = {
        'epochs': [args.num_epochs],
        'sample_dim': [ds.sample_size],
        'batch_size': [1, 16, 64],
        'optimizer': ['adam'],
        'encoding_dim': [16, 32, 64, 128],
        'encoder_activation': ['relu'],
        'decoder_activation': ['linear'],
        'loss': ['cosine_proximity']
    }

    gs = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      n_jobs=-1,
                      verbose=1)
    we_train, we_test, le_train, le_test = split_and_reshape(ds)
    model = gs.fit(we_train, le_train)
    print('Best estimator:')
    print(model.best_estimator_)
    print('Best parameters:')
    print(model.best_params_)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '--word-embeddings-file',
        help='The path to the zip file containing word embeddings.',
        default='/data/corola-word-embeddings.vec.zip')
    parser.add_argument(
        '--lemma-embeddings-file',
        help='The path to the zip file containing lemma embeddings.',
        default='/data/corola-lemma-embeddings.vec.zip')
    parser.add_argument(
        '--word-lemma-pairs-file',
        help='The path to the csv file containing word-lemma pairs.',
        default='/data/word-lemmas.csv')
    parser.add_argument(
        '--save-common-wl-pairs-to',
        help='The path to the file where to save common word-lemma pairs.',
        default=None)
    parser.add_argument('--num-epochs',
                        help='Number of training epochs.',
                        type=int,
                        default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
