import logging
from argparse import ArgumentParser
from input import WordEmebeddingsDataset
from utils import split_and_reshape
from vae import build_vae_model
from vae import build_model_callbacks
from input import DataSample


def run(args):
    logging.info("Loading dataset...")
    ds = WordEmebeddingsDataset(
        word_embeddings_file=args.word_embeddings_file,
        lemma_embeddings_file=args.lemma_embeddings_file,
        word_lemma_pairs_file=args.word_lemma_pairs_file,
        word_inflections_file=args.word_inflections_file,
        data_file=args.data_file)
    ds.initialize()

    logging.info("Building the model...")
    vae, _, _ = build_vae_model(args.intermediate_dim, args.latent_dim,
                                ds.sample_size, not args.reconstruct_word)
    # print(vae.summary())

    logging.info("Start training the model...")
    we_train, we_test, le_train, le_test = split_and_reshape(ds)
    we_train = we_train / 255
    we_test = we_test / 255
    le_train = le_train / 255
    le_test = le_test / 255
    vae.fit(x=we_train,
            y=le_train,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            callbacks=build_model_callbacks(
                tensorboard_log_dir=args.tensorboard_log_dir,
                use_early_stopping=not args.no_early_stopping,
                early_stopping_patience=args.early_stopping_patience),
            shuffle=True)


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
    parser.add_argument('--data-file',
                        help='The path to the data file for the dataset.',
                        default=None)
    parser.add_argument('--num-epochs',
                        help='Number of training epochs.',
                        type=int,
                        default=50)
    parser.add_argument('--batch-size',
                        help="Batch size for training.",
                        default=128)
    parser.add_argument('--latent-dim',
                        help="Number of dimensions of the latent variable.",
                        type=int,
                        default=100)
    parser.add_argument(
        '--intermediate-dim',
        help="Number of dimensions of the intermediate representation.",
        type=int,
        default=512)
    parser.add_argument('-tensorboard-log-dir',
                        help='The root path where to save TensorBoard logs.',
                        default='logs')
    parser.add_argument('--no-early-stopping',
                        help='Remove EarlyStopping callback from the model.',
                        action='store_false')
    parser.add_argument(
        '--early-stopping-patience',
        help='Specifies how many epochs to wait before early stopping.',
        type=int,
        default=2)
    parser.add_argument(
        '--reconstruct-word',
        help='Flag to signal the model to reconstruct word instead of lemma.',
        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
