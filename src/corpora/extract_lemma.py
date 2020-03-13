import csv
import logging
import string
import xml.etree.ElementTree as etree
import zipfile
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

TRANSLATION_TABLE = str.maketrans({p: None for p in string.punctuation})


def normalize(text):
    """
    Transforms the given text to lowercase and removes punctuation.

    Parameters
    ----------
    text: string
        The text to normalize.

    Returns
    -------
    string
        The text after normalization.
    """
    return text.lower().translate(TRANSLATION_TABLE)


def get_word_and_lemma(token):
    """
    Extracts the word and its lemma from token element.

    Parameters
    ----------
    token: xml.etree.ElementTree.Element
        The token xml element that holds the word and lemma.

    Returns
    -------
    tuple of (string, string)
        The word and its associated lemma.
    """
    form_element = token.find('form')
    lemma_element = token.find('lemma')

    return normalize(form_element.text), normalize(lemma_element.text)


def iterate_corpus(corpus_file):
    """
    Opens the corpus zip file and returns an iterator over the contents
    as XML root elements.

    Parameters
    ----------
    corpus_file: string
        The path to the corpus file.

    Returns
    -------
    iterator of <xml.etree.Element>
        The iterator over XML files from the corpus.
    """
    logging.info("Opening corpus file...")
    with zipfile.ZipFile(corpus_file) as corpus:
        for name in corpus.namelist():
            logging.info("Reading contents of file '{}'.".format(name))
            with corpus.open(name) as f:
                try:
                    tree = etree.parse(f)
                    root = tree.getroot()
                    yield root
                except etree.ParseError:
                    message = "File '{}' could not be read due to ParseError."
                    logging.warning(message.format(name))
                    continue


def parse_corpus(corpus_file):
    """
    Extracts words and their lemmas from the corpus.

    Parameters
    ----------
    corpus_file: string
        The path to the zip file containing corpus data.

    Returns
    -------
    iterable of <word, lemma>
        The collection of mappings between words and their lemmas.
    """
    word_lemmas = {}
    for root in iterate_corpus(corpus_file):
        for token in root.iter('token'):
            word, lemma = get_word_and_lemma(token)
            if word and (not word.isnumeric()):
                word_lemmas[word] = lemma
        # with corpus.open(corpus.namelist()[0]) as f:
        #     tree = etree.parse(f)
        #     root = tree.getroot()

    return word_lemmas


def save_word_lemmas(word_lemmas, output_file, separator):
    """
    Writes the word to lemma mappings to the specified file.

    Parameters
    ----------
    word_lemmas: dict of <string, string>
        The dictionary containing the mappings between words and their lemma.
    output_file: string
        The path to the csv file where to save the mappings.
    separator: string
        The delimiter char to use for the CSV file.
    """
    logging.info("Writing mappings to '{}'...".format(output_file))
    with open(output_file, 'wt') as f:
        writer = csv.writer(f, delimiter=separator)
        writer.writerow(['Word', 'Lemma'])
        for word, lemma in word_lemmas.items():
            writer.writerow([word, lemma])
    logging.info("Done.")


def run(args):
    message = """
Computing word to lemma mappings with the following parameters:
    corpus-file: {}
    output-file: {}
"""
    logging.info(message.format(args.corpus_file, args.output_file))
    word_lemmas = parse_corpus(args.corpus_file)
    save_word_lemmas(word_lemmas, args.output_file, args.separator)
    logging.info("That's all folks!")


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--corpus-file',
                        help='The path to the corpus file.',
                        default='/data/corpora/marcell.zip')
    parser.add_argument('--output-file',
                        help='The path to the output CSV file.',
                        default='/output/word-lemmas.csv')
    parser.add_argument(
        '--separator',
        help='The separator character for the output CSV file.',
        default='\t')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
