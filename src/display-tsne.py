import logging
import pickle
import matplotlib.pyplot as plt

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
    logging.info("Loading word to index map...")
    w2idx = load_word_index_map(args.tsne_embeddings_map)

    logging.info("Loading embeddings...")
    points = load_embeddings(args.tsne_embeddings_file)

    plt.clf()
    fig, ax = plt.subplots(figsize=(40, 20))
    logging.info("Building the scatter plot...")
    ax.scatter(x=points[:, 0], y=points[:, 1])
    logging.info("Annotating points...")
    current, total = 0, len(w2idx)
    for lemma, info in w2idx.items():
        current += 1
        if current % 1000 == 0:
            logging.info("Annotating point {}/{}...".format(current, total))
        index = info['index']
        ax.annotate(lemma, (points[index, 0], points[index, 1]))
        for word, index in info['words'].items():
            ax.annotate(word, (points[index, 0], points[index, 1]))
    logging.info("Finished annotating.")
    if args.interactive:
        plt.show()
    else:
        logging.info("Saving plot to {}...".format(args.output_file))
        plt.savefig(args.output_file)
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
    parser.add_argument(
        '--interactive',
        help="If true pyplot will display results instead of saving to file.",
        action='store_true')
    parser.add_argument(
        '--output-file',
        help="Path of the file where to save plot in non-interactive mode.",
        default="tsne-plot.png")
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
