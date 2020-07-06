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

    lbl2idx = {}
    X = []
    for i in range(100):
        word = ds.words[i]
        lemma = ds.lemmas[i]
        we = ds.word_embeddings[i]
        le = ds.lemma_embeddings[i]
        if lemma not in lbl2idx:
            lbl2idx[lemma] = i
            X.append(le)
        lbl2idx[word] = i
        X.append(we)

    logging.info("Running t-SNE with perplexity={}".format(args.perplexity))
    model = TSNE(n_components=2, perplexity=args.perplexity, random_state=2020)
    X = model.fit_transform(X)
    logging.info("Output shape: {}".format(X.shape))
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.scatter(x=X[:, 0], y=X[:, 1])

    for lbl, idx in lbl2idx.items():
        ax.annotate(lbl, (X[idx, 0], X[idx, 1]))
    plt.savefig("tsne-perplexity-{}.png".format(args.perplexity))

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
