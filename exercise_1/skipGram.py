import argparse
import pandas as pd
import numpy as np

from skip_gram_model import SkipGram
from data_loader import *
from tools import *

__authors__ = ['Benoit Laures', 'Ayush Rai', 'Paul Asquin']
__emails__ = ['benoit.laures@student.ecp.fr', 'ayush.rai2512@student-cs.fr', 'paul.asquin@student.ecp.fr']

NUMBER_LINES = 100000
N_EPOCHS = 500
DECAY_INTERVAL = 5
EMBEDDED_SIZE = 300
NEGATIVE_SAMPLING_SIZE = 5
WINDOW_SIZE = 3
LEARNING_RATE = 5e-3
DECAY_FACTOR = 0.99
BATCH_SIZE = 256


def loadPairs(path):
    colomns = ['word1', 'word2', 'similarity']
    data = pd.read_csv(path, delimiter='\t', names=colomns, header=None)
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return list(pairs)


def similarity(word1: str, word2: str, saved_model: dict) -> float:
    """
    Computes the cosine similarity between the 2 given words.

    Parameters
    ----------
    word1, word2 : str
        Input words.
    saved_model : dict
        Loaded model from previous training.

    Returns
    ----------
    out : float
        Cosine similarity.
    """
    try:
        word1_embed = saved_model[word1]
        word2_embed = saved_model[word2]
    except KeyError:  # One word is unknown
        return -1

    return word1_embed.dot(word2_embed) / (np.linalg.norm(word1_embed) * np.linalg.norm(word2_embed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        # sentences = text2sentences(opts.text)
        # print("Loading from", opts.text)
        sentences = read_dataset(path_to_file=opts.text, number_lines=NUMBER_LINES)
        # print('Number of sentences = ', len(sentences))
        word_to_id, id_to_word = map_words(sentences)
        # print('Number of words = ', len(word_to_id))
        x_ids, y_ids, word_frequencies = generate_ids_datasets(sentences, word_to_id, window_size=WINDOW_SIZE)
        x, y = generate_matrices_datasets(x_ids, y_ids, len(word_to_id))
        # print('Training matrices shape = ', x.shape)

        x = x.tocsr()
        y = y.tocsr()

        sg = SkipGram(len(word_to_id), word_frequencies, embed_dim=EMBEDDED_SIZE, id_to_word=id_to_word)
        sg.train(x, y, y_ids, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, neg_sampling_size=NEGATIVE_SAMPLING_SIZE,
                 learning_rate=LEARNING_RATE, decay_factor=DECAY_FACTOR, decay_interval=DECAY_INTERVAL)

        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        dictio = SkipGram.load(opts.model)

        for a, b, _ in pairs:
            print(a, b, similarity(a, b, dictio))
