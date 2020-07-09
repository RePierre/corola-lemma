import logging
from argparse import ArgumentParser
from input import WordEmebeddingsDataset
from sklearn.manifold import TSNE
import pickle


def run(args):
    ds = WordEmebeddingsDataset(
        word_embeddings_file=args.word_embeddings_file,
        lemma_embeddings_file=args.lemma_embeddings_file,
        word_lemma_pairs_file=args.word_lemma_pairs_file,
        word_inflections_file=args.word_inflections_file)
    ds.initialize()

    clusters = {}
    X = []
    j = 0
    for i in range(ds.num_samples):
        word = ds.words[i]
        lemma = ds.lemmas[i]
        we = ds.word_embeddings[i]
        le = ds.lemma_embeddings[i]
        if lemma not in clusters:
            clusters[lemma] = {'index': j, 'words': {}}
            X.append(le)
            j += 1
        clusters[lemma]['words'][word] = j
        X.append(we)
        j += 1

    logging.info("Running t-SNE with perplexity={}".format(args.perplexity))
    model = TSNE(n_components=2, perplexity=args.perplexity, random_state=2020)
    X = model.fit_transform(X)
    logging.info("Output shape: {}".format(X.shape))

    logging.info("Writing embeddings to {}...".format(
        args.tsne_embeddings_file))
    with open(args.tsne_embeddings_file, 'wb') as out_f:
        pickle.dump(X, out_f)

    logging.info("Writing embeddings map to {}...".format(
        args.tsne_embeddings_map))
    with open(args.tsne_embeddings_map, 'wb') as out_f:
        pickle.dump(clusters, out_f)

    logging.info("That's all folks!")


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
        '--word-inflections-file',
        help='The path to the text file containing word inflections.',
        default='/data/word-inflections.txt')
    parser.add_argument('--perplexity',
                        help="Value for perplexity parameter of TSNE.",
                        type=int,
                        default=10)
    parser.add_argument(
        '--tsne-embeddings-file',
        help='Path to the output where to save the post t-SNE embeddings.',
        default='tsne-results.obj')
    parser.add_argument(
        '--tsne-embeddings-map',
        help='Path to the output file mapping embeddings with words.',
        default='tsne-word-map.obj')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
