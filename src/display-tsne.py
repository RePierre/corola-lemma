import logging
import pickle

from argparse import ArgumentParser


def load_embeddings(file_path):
    """
    Read the t-SNE embeddings from provided file.

    Parameters
    ----------
    file_path: string
        The path to the file containing t-SNE embeddings.

    Returns
    -------
    embeddings: array, shape (n_samples, n_components)
        The embeddings read from the file.
    """
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)

    return embeddings


def load_word_index_map(file_path):
    """
    Load the mapings of words to indexes in the embeddings matrix.

    Parameters
    ----------
    file_path: string
        The path to the file containing mappings.

    Returns
    -------
    word_map: dict
        The mapping of words to their indices.
    """
    with open(file_path, 'rb') as f:
        word_map = pickle.load(f)

    return word_map


def run(args):
    w2idx = load_word_index_map(args.tsne_embeddings_map)
    points = load_embeddings(args.tsne_embeddings_file)
    print(w2idx)
    print(points)
    logging.info("That's all folks!")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--tsne-embeddings-file',
                        help='Path to the file containing t-SNE embeddings.',
                        default='tsne-results.obj')
    parser.add_argument(
        '--tsne-embeddings-map',
        help='Path to the output file mapping embeddings with words.',
        default='tsne-word-map.obj')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
