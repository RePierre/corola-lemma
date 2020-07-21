from zipfile import ZipFile
import pandas as pd
import numpy as np
from collections import namedtuple
import csv
import logging
import pickle

DataSample = namedtuple('DataSample',
                        ['word', 'word_embedding', 'lemma', 'lemma_embedding'])


class WordEmebeddingsDataset:
    """
    Represents the dataset of word and lemma embeddings.
    """
    def __init__(self,
                 word_embeddings_file=None,
                 lemma_embeddings_file=None,
                 word_lemma_pairs_file=None,
                 word_inflections_file=None,
                 data_file=None):
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
        word_inflections_file: string
            The path to the file containing valid word inflections.
            This file will be used to filter the list of words and lemmas.
        data_file: string
            The path to the file containing pickled data.
        """
        self._word_embeddings_file = word_embeddings_file
        self._lemma_embeddings_file = lemma_embeddings_file
        self._word_lemma_pairs_file = word_lemma_pairs_file
        self._word_inflections_file = word_inflections_file
        self._data_file = data_file

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

    @property
    def num_samples(self):
        """
        Returns the number of word-lemma pairs from the dataset.
        """
        return len(self._dataset)

    def initialize(self):
        """
        Initializes the data sets by performing a join between word embeddings,
        word-lemma pairs and lemma embeddings.

        Raises
        ------
        AssertionError
            If the sample size of lemma embeddings is not the same as
            the sample size of word embeddings.
        """
        # Maybe start another process to free some memory
        # https://stackoverflow.com/questions/32167386/force-garbage-collection-in-python-to-free-memory
        if self._data_file:
            logging.info("Loading dataset from file {}".format(
                self._data_file))
            with open(self._data_file, 'rb') as data_file:
                self._dataset = pickle.load(data_file)
            self._sample_size = self._dataset[0].word_embedding.shape[0]
        else:
            self._build_dataset()
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

    def save(self, file_path):
        """
        Saves the dataset into the specified file.

        Parameters
        ----------
        file_path: string
            The path to the file where to save the dataset.
        """
        with open(file_path, 'wb') as data_file:
            pickle.dump(self._dataset, data_file)

    def _build_dataset(self):
        logging.info("Loading word inflections...")
        word_forms = self._load_word_inflections()

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
            if self._word_inflections_file and (lemma not in word_forms):
                continue
            lemma_embedding = le_dict[lemma]
            self._dataset.append(
                DataSample(word=word,
                           word_embedding=word_embedding,
                           lemma=lemma,
                           lemma_embedding=lemma_embedding))

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
        return {
            row['Word']: row['Lemma']
            for _, row in pairs_df.iterrows()
            if self._is_valid_word(row['Word'])
            and self._is_valid_word(row['Lemma'])
        }

    def _is_valid_word(self, word: str):
        """
        Determines if a word is valid by the presence of unusual characters.

        Parameters
        ----------
        word: str
            The word to check for validity.

        Returns
        -------
        is_valid: boolean
            Whether the word is valid or not.
        """
        return isinstance(word, str) and np.all(
            [c.isalpha() or c == "'" or c == "-" for c in word])

    def _load_word_inflections(self):
        """
        Loads the set of word inflections if source file is specified.

        Returns
        -------
        word_inflections: set of strings
            The set of word inflections is source file is specified;
        otherwise None.
        """
        if not self._word_inflections_file:
            return None

        word_inflections = set()
        with open(self._word_inflections_file, encoding='UTF-8') as input_file:
            for line in input_file:
                word_inflections.add(line.strip().lower())

        return word_inflections


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    ds = WordEmebeddingsDataset(
        word_embeddings_file='/data/corola-word-embeddings.vec.zip',
        lemma_embeddings_file='/data/corola-lemma-embeddings.vec.zip',
        word_lemma_pairs_file='/data/word-lemmas.csv',
        word_inflections_file='/data/word-inflections.txt')
    ds.initialize()
    ds.save('/data/dataset.data')
    ds = WordEmebeddingsDataset(data_file='/data/dataset.data')
    ds.initialize()
    for i in range(10):
        print(ds._dataset[i])
    # ds.save_word_lemma_pairs('pairs.csv')
    logging.info("That's all folks!")
