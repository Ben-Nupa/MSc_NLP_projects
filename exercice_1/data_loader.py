import re
from typing import List, Tuple
import numpy as np
from scipy import int8
from scipy.sparse import lil_matrix

REDUCE_FLOAT = False


def text2sentences(path: str, nb_lines=100) -> List[List[str]]:
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            line = re.sub('-', ' ', line)  # Replace '-' by ' '
            line = re.sub(r'[^\w\s]', '', line)  # Remove punctuation
            sentences.append(line.lower().split())
            if len(sentences) == nb_lines:
                break
    return sentences


def map_words(sentences: List[List[str]]) -> Tuple[dict, dict]:
    """
    Maps the given words to their IDs in a dictionary and reciprocally.

    Parameters
    ----------
    sentences : list[list[str]]
        List of sentences where sentences are list of words.

    Returns
    ----------
    out : tuple[dict, dict]
        Tuple of dictionaries : {word --> id} and {id --> word}.
    """
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
    """
    Generates the datasets as IDs lists for the skip-gram models where input are center words and output are context
    words. The i-th element of the x list corresponds to a j-th center word and the i-th element of the y list
    corresponds to a context word of the j-th center word. Also computes the frequencies list for each word, sorted by
    word IDs.

    Parameters
    ----------
    sentences : list
        List of sentences where sentences are list of words.
    word_to_id : dict
        Dictionary mapping a word to its ID.
    window_size : int
        Size for the context words.

    Returns
    ----------
    out : tuple[list, list, ndarray]
        Input list of center word IDs: [idx_0, idx_0, ..., idx_1, idx_1, ...]
        Output list of context word IDs: [idx_00, idx_01, ..., idx_10, idx_11, ...]
        Frequency of each word in the same order as words ids.
    """
    x = []
    y = []
    word_frequencies = np.zeros(len(word_to_id))
    for sentence in sentences:
        for i, center_word in enumerate(sentence):
            word_frequencies[word_to_id[center_word]] += 1
            for j in range(max(0, i - window_size), min(len(sentence) - 1, i + window_size + 1)):
                if j == i:
                    continue
                context_word = sentence[j]
                x.append(word_to_id[center_word])
                y.append(word_to_id[context_word])
    return x, y, word_frequencies


def generate_matrices_datasets(x_ids: List[int], y_ids: List[int], vocab_size: int) -> Tuple[lil_matrix, lil_matrix]:
    """
    Generates the datasets as sparse one-hot encoded matrices for the skip-gram models where input are center words and
    output are context words. The i-th row of the x array corresponds to a j-th center word and the i-th row of the y
    array corresponds to a context word of the j-th center word.

    See the method 'generate_ids_datasets' for the argument of this method.

    Parameters
    ----------
    x_ids : list[int]
        IDs of input words
    y_ids : list[int]
        IDs of output words.
    vocab_size : int
        Size of the vocabulary (total number of unique words).

    Returns
    ----------
    out : tuple[lil_matrix, lil_matrix]
        Sparse one-hot encoded matrix of the input.
        Sparse one-hot encoded matrix of the output.
    """
    nb_pairs = len(x_ids)

    if REDUCE_FLOAT:
        x = lil_matrix((nb_pairs, vocab_size), dtype=int8)
        y = lil_matrix((nb_pairs, vocab_size), dtype=int8)
    else:
        x = lil_matrix((nb_pairs, vocab_size))
        y = lil_matrix((nb_pairs, vocab_size))

    for i in range(nb_pairs):
        x[i, x_ids[i]] = 1
        y[i, y_ids[i]] = 1
    return x, y
