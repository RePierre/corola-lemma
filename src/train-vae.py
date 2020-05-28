import logging
from argparse import ArgumentParser
from input import WordEmebeddingsDataset
from utils import split_and_reshape
from vae.modelfactory import build_vae_model


def run(args):
    logging.info("Loading dataset...")
    ds = WordEmebeddingsDataset(
        word_embeddings_file=args.word_embeddings_file,
        lemma_embeddings_file=args.lemma_embeddings_file,
        word_lemma_pairs_file=args.word_lemma_pairs_file)
    ds.initialize()

    logging.info("Building the model...")
    vae, _, _ = build_vae_model(int(ds.sample_size / 2), args.latent_dim,
                                ds.sample_size)
    print(vae.summary())

    logging.info("Start training the model...")
    we_train, we_test, le_train, le_test = split_and_reshape(ds)
    vae.fit(x=we_train,
            y=le_train,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
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
    parser.add_argument('--num-epochs',
                        help='Number of training epochs.',
                        type=int,
                        default=50)
    parser.add_argument('--batch-size',
                        help="Batch size for training.",
                        default=128)
    parser.add_argument('--latent_dim',
                        help="Number of dimensions of the latent variable.",
                        type=int,
                        default=10)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
