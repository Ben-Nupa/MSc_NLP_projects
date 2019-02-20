from __future__ import division
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time

from exercice_1.skip_gram_model import SkipGram
from exercice_1.data_loader import *

# useful stuff
import numpy as np
# from scipy.special import expit
from sklearn.preprocessing import normalize

__authors__ = ['author1', 'author2', 'author3']
__emails__ = ['fatherchristmoas@northpole.dk', 'toothfairy@blackforest.no', 'easterbunny@greenfield.de']


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


if __name__ == '__main__':
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
    sentences = text2sentences(PATH_TO_DATA + '/news.en-00001-of-00100')
    print(np.shape(sentences))
    word_to_id, id_to_word = map_words(sentences)
    print(len(word_to_id))
    x_ids, y_ids, word_frequencies = generate_ids_datasets(sentences, word_to_id, window_size=3)
    x, y = generate_matrices_datasets(x_ids, y_ids, len(word_to_id))
    print(x.shape)

    x = x.toarray()
    y = y.toarray()

    begin = time.time()

    # Verify the gradients
    # comparison_sg = SkipGram(len(word_to_id), word_frequencies, 100)
    # comparison_sg.w1 = np.random.uniform(-0.01, 0.01, size=comparison_sg.w1.shape)
    # comparison_sg.w2 = np.random.uniform(-0.01, 0.01, size=comparison_sg.w2.shape)
    # comparison_sg.compare_gradients(x, y, y_ids, 1e-5)

    sg = SkipGram(len(word_to_id), word_frequencies, 100)
    sg.train(x, y, y_ids)

    # sg = SkipGram2(len(word_to_id), 100)
    # sg.train(x_ids, y_ids)

    # plt.show()

    print('END = ', time.time() - begin)
