import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from skip_gram_model import SkipGram
from data_loader import *
from tools import *

__authors__ = ['Benoit Laures', 'Ayush Rai', 'Paul Asquin']
__emails__ = ['benoit.laures@student.ecp.fr', 'ayush.rai2512@student-cs.fr', 'paul.asquin@student.ecp.fr']

NUMBER_LINES = 1000
N_EPOCHS = 50
DECAY_INTERVAL = 1
EMBEDDED_SIZE = 200
WINDOW_SIZE = 3
NEGATIVE_SAMPLING = 3
LEARNING_RATE = 5e-3
BATCH_SIZE = 256


def loadPairs(path):
    colomns = ['word1', 'word2', 'similarity']
    data = pd.read_csv(path, delimiter='\t', names=colomns, header=None)
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return list(pairs)


def similarity(word1, word2, dictio):
    try:
        word1_embed = dictio[word1]
        word2_embed = dictio[word1]
    except KeyError:
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
        print("Loading from", opts.text)
        sentences = read_dataset(path_to_dataset_folder=opts.text, number_lines=NUMBER_LINES)
        print('Number of sentences = ', len(sentences))
        word_to_id, id_to_word = map_words(sentences)
        print('Number of words = ', len(word_to_id))
        x_ids, y_ids, word_frequencies = generate_ids_datasets(sentences, word_to_id, window_size=3)
        x, y = generate_matrices_datasets(x_ids, y_ids, len(word_to_id))
        print('Training matrices shape = ', x.shape)

        x = x.tocsr()
        y = y.tocsr()

        sg = SkipGram(len(word_to_id), word_frequencies, embed_dim=EMBEDDED_SIZE, id_to_word=id_to_word)
        sg.train(x, y, None, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, neg_sampling_size=NEGATIVE_SAMPLING,
                 learning_rate=LEARNING_RATE, decay_factor=0.99, decay_interval=DECAY_INTERVAL)

        np.save('models/' + str(N_EPOCHS) + '_' + str(NUMBER_LINES) + '_' + str(EMBEDDED_SIZE) + '_w1.npy', sg.w1)
        np.save('models/' + str(N_EPOCHS) + '_' + str(NUMBER_LINES) + '_' + str(EMBEDDED_SIZE) + '_w2.npy', sg.w2)
        # sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)
        dictio = SkipGram.load(opts.model)

        for a, b, _ in pairs:
            print(a, b, similarity(a, b, dictio))

    if False:
        # PATH_TO_DATA = 'data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'
        # Loading sentences
        sentences = read_dataset(path_to_dataset_folder=PATH_TO_DATA, number_lines=NUMBER_LINES)
        # print(sentences)

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

        # sg = SkipGram(len(word_to_id), word_frequencies, embed_dim=50)
        # sg.train(x, y, y_ids, n_epochs=N_EPOCHS, batch_size=16, neg_sampling_size=5, learning_rate=1e-1, decay_factor=0.99,
        #          decay_interval=DECAY_INTERVAL, save_model_every_n_epochs=SAVE_MODEL_EVERY_N_EPOCHS)

        print('END = ', time.time() - begin, "seconds")

    plt.show()
