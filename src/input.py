from zipfile import ZipFile
import pandas as pd
import numpy as np


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
    def word_embeddings(self):
        """
        Returns the list of word embeddings.
        """
        return [we for we in self._dataset['WordEmbedding']]

    @property
    def lemma_embeddings(self):
        """
        Returns the list of lemma embeddings.
        """
        return [le for le in self._dataset['LemmaEmbedding']]

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

        pairs_df = pd.read_csv(self._word_lemma_pairs_file, sep='\t', header=0)

        we_df, we_size = self._load_embeddings_dataframe(
            self._word_embeddings_file, ['Word', 'WordEmbedding'])
        le_df, le_size = self._load_embeddings_dataframe(
            self._lemma_embeddings_file, ['Lemma', 'LemmaEmbedding'])

        if we_size == le_size:
            self._sample_size = we_size
        else:
            raise AssertionError(
                'Word embeddings size does not match lemma embeddings size.')

        common_df1 = we_df.merge(pairs_df, left_on='Word', right_on='Word')
        self._dataset = common_df1.merge(le_df,
                                         left_on='Lemma',
                                         right_on='Lemma')
        if with_preview:
            print('Word-lemma pairs:')
            print(pairs_df.head())
            print('Word vectors:')
            print(we_df.head())
            print('Lemma vectors:')
            print(le_df.head())
            print('First merge:')
            print(common_df1.head())
            print('Second merge:')
            print(self._dataset.tail())

    def save_word_lemma_pairs(self, file_path):
        """
        Saves the word-lemma pairs from dataset to the specified file.

        Parameters
        ----------
        file_path: string
            The path to the csv file where to save the word-lemma pairs.
        """
        self._dataset.to_csv(file_path, columns=['Word', 'Lemma'])

    def _load_embeddings_dataframe(self, file_name, columns):
        embeddings = self._load_embeddings(file_name)
        embeddings = list(embeddings)
        _, e = embeddings[0]
        sample_size = e.shape[0]

        df = pd.DataFrame.from_records(embeddings, columns)
        return df, sample_size

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


if __name__ == '__main__':
    # Iterate over input files and build a set
    # containing the intersection of words and lemmas.
    # Afterwards, load the embeddings based on that set.
    # load_word_embeddings('/data/corola-word-embeddings.vec.zip')
    # load_lemma_embeddings('/data/corola-lemma-embeddings.vec.zip')
    # load_word_lemma_pairs('/data/word-lemmas.csv')
    ds = WordEmebeddingsDataset(
        word_embeddings_file='/data/corola-word-embeddings.vec.zip',
        lemma_embeddings_file='/data/corola-lemma-embeddings.vec.zip',
        word_lemma_pairs_file='/data/word-lemmas.csv')
    ds.initialize()
