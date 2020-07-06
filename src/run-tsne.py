import logging
from argparse import ArgumentParser
from input import WordEmebeddingsDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
    for i in range(50):
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

    for lemma in clusters.keys():
        plt.clf()
        x, y, labels = [], [], []
        index = clusters[lemma]['index']
        x.append(X[index, 0])
        y.append(X[index, 1])
        labels.append(lemma)
        for word, index in clusters[lemma]['words'].items():
            x.append(X[index, 0])
            y.append(X[index, 1])
            labels.append(word)
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.scatter(x=x, y=y)
        for lbl, x_coord, y_coord in zip(labels, x, y):
            ax.annotate(lbl, (x_coord, y_coord))
        plt.savefig("{}-tsne-perplexity-{}.png".format(lemma, args.perplexity))

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
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
