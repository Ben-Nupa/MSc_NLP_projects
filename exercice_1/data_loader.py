import numpy as np
import re
from scipy.sparse import lil_matrix
from typing import List, Tuple


def text2sentences(path: str) -> List[List[str]]:
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            line = re.sub('-', ' ', line)  # Replace '-' by ' '
            line = re.sub(r'[^\w\s]', '', line)  # Remove punctuation
            sentences.append(line.lower().split())
            print(sentences[-1])
            break
    return sentences


def map_words(sentences: List[List[str]]) -> Tuple[dict, dict]:
    idx_word = 0
    word_to_id = {}
    id_to_word = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_id.keys():
                word_to_id[word] = idx_word
                id_to_word[idx_word] = word
                idx_word += 1
    return word_to_id, id_to_word


def generate_ids_datasets(sentences: List[List[str]], word_to_id: dict, window_size=3) -> Tuple[list, list, np.ndarray]:
    x = []
    y = []
    word_frequencies = np.zeros(len(word_to_id))
    for sentence in sentences:
        for i, center_word in enumerate(sentence):
            word_frequencies[word_to_id[center_word]] += 1
            for j in range(max(0, i - window_size), min(len(sentence) - 1, i + window_size)):
                if j == i:
                    continue
                context_word = sentence[j]
                x.append(word_to_id[center_word])
                y.append(word_to_id[context_word])
    return x, y, word_frequencies


def generate_matrices_datasets(x_ids: List[int], y_ids: List[int], vocab_size: int) -> Tuple[lil_matrix, lil_matrix]:
    nb_pairs = len(x_ids)
    x = lil_matrix((nb_pairs, vocab_size))
    y = lil_matrix((nb_pairs, vocab_size))
    for i in range(nb_pairs):
        x[i, x_ids[i]] = 1
        y[i, y_ids[i]] = 1
    return x, y
