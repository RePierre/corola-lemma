import logging
from argparse import ArgumentParser
from input import WordEmebeddingsDataset
from gridsearch.modelfactory import build_autoencoder
from sklearn.model_selection import GridSearchCV
from utils import split_and_reshape


def run(args):
    logging.info("Loading dataset...")
    ds = WordEmebeddingsDataset(
        word_embeddings_file=args.word_embeddings_file,
        lemma_embeddings_file=args.lemma_embeddings_file,
        word_lemma_pairs_file=args.word_lemma_pairs_file)
    ds.initialize()
    if args.save_common_wl_pairs_to:
        ds.save_word_lemma_pairs(args.save_common_wl_pairs_to)

    logging.info("Building model...")
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

    logging.info("Start searching for parameters...")
    we_train, we_test, le_train, le_test = split_and_reshape(ds)
    model = gs.fit(we_train, le_train)
    print('Best estimator:')
    print(model.best_estimator_)
    print('Best parameters:')
    print(model.best_params_)
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
        '--save-common-wl-pairs-to',
        help='The path to the file where to save common word-lemma pairs.',
        default=None)
    parser.add_argument('--num-epochs',
                        help='Number of training epochs.',
                        type=int,
                        default=100)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
