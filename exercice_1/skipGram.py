SAVE_MODEL_EVERY_N_EPOCHS = 100
NUMBER_LINES = 100
N_EPOCHS = 100
DECAY_INTERVAL = 10

# from __future__ import division
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
import glob

from exercice_1.skip_gram_model import SkipGram
from exercice_1.data_loader import *
from exercice_1.tools import *

# useful stuff
import numpy as np
# from scipy.special import expit
from sklearn.preprocessing import normalize

__authors__ = ['Benoit Laures', 'Ayush Rai', 'Paul Asquin']
__emails__ = ['benoit.laures@student.ecp.fr', 'ayush.rai2512@student-cs.fr', 'paul.asquin@student.ecp.fr']


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--text', help='path containing training data', required=True)
    # parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    # parser.add_argument('--test', help='enters test mode', action='store_true')
    #
    # opts = parser.parse_args()
    #
    # if not opts.test:
    #     sentences = text2sentences(opts.text)
    #     sg = SkipGram(sentences)
    #     sg.train(...)
    #     sg.save(opts.model)
    #
    # else:
    #     pairs = loadPairs(opts.text)
    #
    #     sg = SkipGram.load(opts.model)
    #     for a, b, _ in pairs:
    #         print(sg.similarity(a, b))

    PATH_TO_DATA = 'data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'

    # Loading sentences
    # sentences = text2sentences(PATH_TO_DATA + '/news.en-00001-of-00100', nb_lines=10)
    sentences = read_dataset(path_to_dataset_folder=PATH_TO_DATA, number_lines=NUMBER_LINES)
    print('Number of sentences = ', len(sentences))
    word_to_id, id_to_word = map_words(sentences)
    print('Number of words = ', len(word_to_id))
    x_ids, y_ids, word_frequencies = generate_ids_datasets(sentences, word_to_id, window_size=3)
    x, y = generate_matrices_datasets(x_ids, y_ids, len(word_to_id))
    print('Training matrices shape = ', x.shape)

    # Transform to CSR sparse matrix for memory issues
    x = x.tocsr()
    y = y.tocsr()
    # x = x.toarray().astype(np.int8)
    # y = y.toarray().astype(np.int8)

    begin = time.time()

    # Verify the gradients
    # comparison_sg = SkipGram(len(word_to_id), word_frequencies, 100)
    # comparison_sg.w1 = np.random.uniform(-0.01, 0.01, size=comparison_sg.w1.shape)
    # comparison_sg.w2 = np.random.uniform(-0.01, 0.01, size=comparison_sg.w2.shape)
    # comparison_sg.compare_gradients(x, y, None, 1e-5)

    sg = SkipGram(len(word_to_id), word_frequencies, 100)
    sg.train(x, y, y_ids, n_epochs=N_EPOCHS, batch_size=64, neg_sampling_size=5, learning_rate=1e-2, decay_factor=0.99,
             decay_interval=DECAY_INTERVAL, save_model_every_n_epochs=SAVE_MODEL_EVERY_N_EPOCHS)

    print('END = ', time.time() - begin)

    plt.show()


main()
