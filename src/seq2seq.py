from argparse import ArgumentParser
import numpy as np
import logging
from tensorflow import keras


def get_text_metadata(text):
    alphabet = set()
    max_len = 0
    for line in text:
        alphabet = alphabet.union(set([c for c in line]))
        max_len = max(max_len, len(line))
    alphabet = alphabet.union(set(' '))
    return sorted(list(alphabet)), max_len


def build_char_maps(alphabet):
    char_map = {c: i for i, c in enumerate(alphabet)}
    inv_char_map = {i: c for i, c in enumerate(alphabet)}
    return char_map, inv_char_map


def build_seq2seq_model(num_chars, num_decoder_tokens, num_latent_dimensions):
    """Builds the Sequence to Sequence model.

    Parameters
    ----------
    num_chars: integer
        The number of dimensions used to represent one character.
    num_latent_dimensions: integer
        The number of dimmensions for hidden layers of the recurrent
        network.

    Returns
    -------
    keras.Model
        The compiled sequence to sequence model.
    """
    encoder_input = keras.Input(shape=(None, num_chars), name="encoder-input")
    encoder = keras.layers.LSTM(num_latent_dimensions,
                                return_state=True,
                                name="encoder-lstm")
    _, h, c = encoder(encoder_input)
    # Discard output and keep states
    encoder_states = [h, c]

    decoder_input = keras.Input(shape=(None, num_decoder_tokens),
                                name="decoder-input")
    decoder_rec_layer = keras.layers.LSTM(num_latent_dimensions,
                                          return_sequences=True,
                                          return_state=True,
                                          name="decoder-lstm")
    decoder_output, _, _ = decoder_rec_layer(decoder_input,
                                             initial_state=encoder_states)
    dense = keras.layers.Dense(num_decoder_tokens,
                               activation='softmax',
                               name="dense-softmax")
    decoder_output = dense(decoder_output)
    model = keras.Model([encoder_input, decoder_input], decoder_output)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_model(model_path, num_latent_dimensions):
    """Loads the trained model and splits it into encoder and decoder.

    Parameters
    ----------
    model_path: string
        The path to the directory where model was saved.
    num_latent_dimensions: integer
        Number of dimensions of hidden layers.

    Returns
    -------
    (encoder, decoder)
        A tuple of keras.Model instances representing the encoder
        and the decoder.
    """
    model = keras.models.load_model(model_path)
    encoder_input = model.input[0]
    _, encoder_h, encoder_c = model.layers[2].output
    encoder_states = [encoder_h, encoder_c]
    encoder = keras.Model(encoder_input, encoder_states)

    decoder_input = model.input[1]
    decoder_h_in = keras.Input(shape=(num_latent_dimensions, ),
                               name="decoder_h_in")
    decoder_c_in = keras.Input(shape=(num_latent_dimensions, ),
                               name="decoder_c_in")
    decoder_input_states = [decoder_h_in, decoder_c_in]
    decoder_rec_layer = model.layers[3]
    decoder_output, decoder_h_out, decoder_c_out = decoder_rec_layer(
        decoder_input, initial_state=decoder_input_states)
    decoder_states = [decoder_h_out, decoder_c_out]
    dense = model.layers[4]
    decoder_output = dense(decoder_output)
    decoder = keras.Model([decoder_input] + decoder_input_states,
                          [decoder_output] + decoder_states)
    return encoder, decoder


def build_encoder_input(text, char_map, max_seq_len):
    X = np.zeros((len(text), max_seq_len, len(char_map)), dtype="float32")
    for i, line in enumerate(text):
        for j, char in enumerate(line):
            X[i, j, char_map[char]] = 1.0
        X[i, j + 1:, char_map[' ']] = 1.0
    return X


def build_decoder_input(text, char_map, max_seq_len):
    result = np.zeros((len(text), max_seq_len, len(char_map)), dtype="float32")
    for i, line in enumerate(text):
        for j, char in enumerate(line):
            result[i, j, char_map[char]] = 1.0
        result[i, j + 1:, char_map[' ']] = 1.0
    return result


def build_decoder_target(text, char_map, max_seq_len):
    result = np.zeros((len(text), max_seq_len, len(char_map)), dtype="float32")
    for i, line in enumerate(text):
        for j, char in enumerate(line):
            if j > 0:
                result[i, j - 1, char_map[char]] = 1.0
        result[i, j:, char_map[' ']] = 1.0
    return result


def load_input(from_console, file_name):
    if from_console:
        while True:
            input_seq = input("Input: ")
            yield input_seq.strip()
            yn = input("Another? [y/N]: ")
            if (not yn) or (yn.lower() == 'n'):
                break
    else:
        with open(file_name, 'rt', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            yield line.strip()


def read_training_data(file_path, num_samples):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.read().split("\n")

    input_texts = []
    target_texts = []
    for line in lines[:min(num_samples, len(lines) - 1)]:
        input_text, target_text, _ = line.split("\t")
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)

    return input_texts, target_texts


def test_model(args):
    input_texts, target_texts = read_training_data(args.data_path,
                                                   args.num_samples)
    input_characters, max_encoder_seq_length = get_text_metadata(input_texts)
    input_map, inv_input_map = build_char_maps(input_characters)

    target_characters, max_decoder_seq_length = get_text_metadata(target_texts)
    target_map, inv_target_map = build_char_maps(target_characters)
    num_decoder_tokens = len(target_characters)

    encoder, decoder = load_model(args.load_model_from, args.latent_dim)
    for line in load_input(args.keyboard_input, args.input_file):
        encoded_line = build_encoder_input([line], input_map,
                                           max_encoder_seq_length)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_map['\t']] = 1.0

        stop_condition = False
        prediction = ""
        encoder_state = encoder.predict(encoded_line)
        while not stop_condition:
            output_tokens, h, c = decoder.predict([target_seq] + encoder_state)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = inv_target_map[sampled_token_index]
            prediction += sampled_char

            if sampled_char == '\n' or len(
                    prediction) > max_decoder_seq_length:
                stop_condition = True
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0
            encoder_state = [h, c]
        print("Prediction: {}".format(prediction))


def train_model(args):
    input_texts, target_texts = read_training_data(args.data_path,
                                                   args.num_samples)
    input_characters, max_encoder_seq_length = get_text_metadata(input_texts)
    input_map, inv_input_map = build_char_maps(input_characters)
    num_encoder_tokens = len(input_characters)

    target_characters, max_decoder_seq_length = get_text_metadata(target_texts)
    target_map, inv_target_map = build_char_maps(target_characters)
    num_decoder_tokens = len(target_characters)

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)

    encoder_input_data = build_encoder_input(input_texts, input_map,
                                             max_encoder_seq_length)
    decoder_input_data = build_decoder_input(target_texts, target_map,
                                             max_decoder_seq_length)
    decoder_target_data = build_decoder_target(target_texts, target_map,
                                               max_decoder_seq_length)

    model = build_seq2seq_model(num_encoder_tokens, num_decoder_tokens,
                                args.latent_dim)
    model.summary()
    print("Shape of encoder input data: {}".format(encoder_input_data.shape))
    print("Shape of decoder input data: {}".format(decoder_input_data.shape))
    print("Shape of decoder target data: {}".format(decoder_target_data.shape))
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        validation_split=0.2,
    )

    model.save(args.save_model_to)


def parse_arguments():
    root_parser = ArgumentParser()

    root_parser.add_argument('--data-path', default="fra.txt")
    root_parser.add_argument('--num_samples', type=int, default=10000)

    subparsers = root_parser.add_subparsers()

    train_model_parser = subparsers.add_parser('train',
                                               help="Train the model.")
    train_model_parser.set_defaults(func=train_model)
    train_model_parser.add_argument(
        '--word-embeddings-file',
        help='The path to the zip file containing word embeddings.',
        default='/data/corola-word-embeddings.vec.zip')
    train_model_parser.add_argument(
        '--lemma-embeddings-file',
        help='The path to the zip file containing lemma embeddings.',
        default='/data/corola-lemma-embeddings.vec.zip')
    train_model_parser.add_argument(
        '--word-lemma-pairs-file',
        help='The path to the csv file containing word-lemma pairs.',
        default='/data/word-lemmas.csv')
    train_model_parser.add_argument(
        '--word-inflections-file',
        help='The path to the text file containing word inflections.',
        default='/data/word-inflections.txt')
    train_model_parser.add_argument(
        '--data-file',
        help='The path to the data file for the dataset.',
        default=None)
    train_model_parser.add_argument('--num-epochs',
                                    help='Number of training epochs.',
                                    type=int,
                                    default=100)
    train_model_parser.add_argument('--batch-size',
                                    help="Batch size for training.",
                                    default=64)
    train_model_parser.add_argument(
        '--latent-dim',
        help="Number of dimensions of the latent hidden layer.",
        type=int,
        default=256)
    train_model_parser.add_argument(
        '--save-model-to',
        help="Path to the directory where to save the model.",
        default='./seq2seq-model')

    test_model_parser = subparsers.add_parser(
        'test', help="Tests the predictions of the trained model.")
    test_model_parser.set_defaults(func=test_model)
    test_model_parser.add_argument('--load-model-from',
                                   help="Directory containing saved model.",
                                   default="./seq2seq-model")
    test_model_parser.add_argument(
        '--latent-dim',
        help="Number of dimensions of the latent hidden layer.",
        type=int,
        default=100)
    test_model_parser.add_argument(
        '--keyboard-input',
        help="Signals that the input should be read from keyboard.",
        action='store_true')
    test_model_parser.add_argument(
        '--input-file',
        help="The path to the input file containing test data.")
    return root_parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    args.func(args)
    logging.info("That's all folks!")
