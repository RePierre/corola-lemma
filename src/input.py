from zipfile import ZipFile
import pandas as pd
import numpy as np
from collections import namedtuple
import csv
import logging

DataSample = namedtuple('DataSample',
                        ['word', 'word_embedding', 'lemma', 'lemma_embedding'])


class WordEmebeddingsDataset:
    """
    Represents the dataset of word and lemma embeddings.
    """
    def __init__(self, word_embeddings_file, lemma_embeddings_file,
                 word_lemma_pairs_file):
        """
        Initializes an instance of WordEmebeddingsDataset class.

        Parameters
        ----------
        word_embeddings_file: string
            The path to the zip file containing word embeddings.
        lemma_embeddings_file: string
            The path to the zip file containing lemma embeddings.
        word_lemma_pairs_file: string
            The path to the CSV file containing word-lemma pairs.
        """
        self._word_embeddings_file = word_embeddings_file
        self._lemma_embeddings_file = lemma_embeddings_file
        self._word_lemma_pairs_file = word_lemma_pairs_file

    @property
    def words(self):
        """
        Returns the list of words.
        """
        return [s.word for s in self._dataset]

    @property
    def word_embeddings(self):
        """
        Returns the list of word embeddings.
        """
        return [s.word_embedding for s in self._dataset]

    @property
    def lemmas(self):
        """
        Returns the list of lemmas.
        """
        return [s.lemma for s in self._dataset]

    @property
    def lemma_embeddings(self):
        """
        Returns the list of lemma embeddings.
        """
        return [s.lemma_embedding for s in self._dataset]

    @property
    def sample_size(self):
        """
        Returns the size of word and lemma embeddings.
        """
        return self._sample_size

    def initialize(self, with_preview=False):
        """
        Initializes the data sets by performing a join between word embeddings,
        word-lemma pairs and lemma embeddings.

        Parameters
        ----------
        with_preview: bool, optional
            If set to true will print previews of the loaded data sets.
            Default is False.

        Raises
        ------
        AssertionError
            If the sample size of lemma embeddings is not the same as
            the sample size of word embeddings.
        """
        # Maybe start another process to free some memory
        # https://stackoverflow.com/questions/32167386/force-garbage-collection-in-python-to-free-memory
        logging.info("Loading word-lemma pairs...")
        pairs_dict = self._load_word_lemma_pairs()
        logging.info("Loading word embeddings...")
        we_dict = self._load_embeddings_dict(self._word_embeddings_file)
        logging.info("Loading lemma embeddings...")
        le_dict = self._load_embeddings_dict(self._lemma_embeddings_file)
        logging.info("Determining sample size...")
        self._determine_sample_size(we_dict, le_dict)

        logging.info("Determining the common words...")
        # Intersect word sets to get common words
        common_words = pairs_dict.keys() & we_dict.keys()
        logging.info("Building the dataset...")
        self._dataset = []
        for word in common_words:
            word_embedding = we_dict[word]
            lemma = pairs_dict[word]
            if lemma not in le_dict:
                continue
            lemma_embedding = le_dict[lemma]
            self._dataset.append(
                DataSample(word=word,
                           word_embedding=word_embedding,
                           lemma=lemma,
                           lemma_embedding=lemma_embedding))
        logging.info("Done. Total samples: {}".format(len(self._dataset)))

    def save_word_lemma_pairs(self, file_path):
        """
        Saves the word-lemma pairs from dataset to the specified file.

        Parameters
        ----------
        file_path: string
            The path to the csv file where to save the word-lemma pairs.
        """
        with open(file_path, 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['Word', 'Lemma'])
            for s in self._dataset:
                writer.writerow([s.word, s.lemma])

    def _determine_sample_size(self, word_embeddings, lemma_embeddings):
        """
        Determines the dimensionality of word and lemma embeddings.

        Raises
        ------
        AssertionError
            If the sample size of lemma embeddings is not the same as
            the sample size of word embeddings.
        """
        we = next(iter(word_embeddings.values()))
        le = next(iter(lemma_embeddings.values()))
        we_dim = we.shape[0]
        le_dim = le.shape[0]
        if we_dim != le_dim:
            raise AssertionError(
                'Word embeddings size does not match lemma embeddings size.')

        self._sample_size = we_dim

    def _load_embeddings_dict(self, file_name):
        """
        Loads words and their embeddings from the specified file.

        Parameters
        ----------
        file_name: string
            The path to the file where word embeddings are stored.

        Returns
        -------
        dict of <word, embedding>
            The dictionary of words and their embeddings.
        """
        return {word: vec for word, vec in self._load_embeddings(file_name)}

    def _load_embeddings(self, zip_file_path):
        with ZipFile(zip_file_path) as input_file:
            name = input_file.namelist()[0]
            with input_file.open(name) as f:
                next(f)  # Skip header line
                for line in f:
                    word, vec = self._parse_line(line)
                    yield word, vec

    def _parse_line(self, line):
        """
        Parses a line into a word and its associated embedding.

        Parameters
        ----------
        line: string
            The line to be parsed.
        """
        parts = line.split()
        word = parts[0].decode('UTF-8')
        vec = np.array(parts[1:], np.float32)
        return word, vec

    def _load_word_lemma_pairs(self):
        """
        Loads the word-lemma pairs as a dictionary.

        Returns
        -------
        dict of <word, lemma>
            The dictionary of word and their associated lemmas.
        """
        pairs_df = pd.read_csv(self._word_lemma_pairs_file, sep='\t', header=0)
        return {row['Word']: row['Lemma'] for _, row in pairs_df.iterrows()}


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    ds = WordEmebeddingsDataset(
        word_embeddings_file='/data/corola-word-embeddings.vec.zip',
        lemma_embeddings_file='/data/corola-lemma-embeddings.vec.zip',
        word_lemma_pairs_file='/data/word-lemmas.csv')
    ds.initialize()
    for i in range(10):
        print(ds._dataset[i])
    ds.save_word_lemma_pairs('pairs.csv')
    logging.info("That's all folks!")
