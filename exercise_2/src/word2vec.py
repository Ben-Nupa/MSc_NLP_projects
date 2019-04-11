import io
import numpy as np
import spacy


class PreTrainedWord2Vec:
    def __init__(self, fname: str, nmax=150000, parser='en'):
        self.word2vec = {}
        self.load_wordvec(fname, nmax)
        self.word2id = {w: i for i, w in enumerate(self.word2vec.keys())}
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embeddings = np.array(list(self.word2vec.values()))
        self.parser = spacy.load(parser)
        self.dimension = -1

    def load_wordvec(self, fname: str, nmax: int):
        """Load the Word2Vec weights of the given file into class variables. Maps each words to an id."""
        self.word2vec = {}
        with io.open(fname, encoding='utf-8') as file:
            next(file)
            for i, line in enumerate(file):
                word, vec = line.split(' ', 1)
                self.word2vec[word] = np.fromstring(vec, sep=' ')
                if i == (nmax - 1):
                    break
        self.dimension = self.word2vec[word].shape[0]
        print('Loaded %s pretrained word vectors of dimension %s' % (len(self.word2vec),  self.dimension))

    @staticmethod
    def sentence_treated(sentence: str) -> str:
        """Change the sentence construction to be more easily manageable and understandable."""
        sentence = sentence.lower()
        sentence = sentence.replace("'s", ' is')
        sentence = sentence.replace("n't", ' not')
        sentence = sentence.replace("'re", ' are')
        sentence = sentence.replace("'m", ' am')
        sentence = sentence.replace("'ve", ' have')
        sentence = sentence.replace("'ll", ' will')
        sentence = sentence.replace("'d", ' would')
        sentence = sentence.replace("-", ' ')
        sentence = sentence.replace("!", ' ! ')
        sentence = sentence.replace(".", ' . ')
        sentence = sentence.replace(":", ' : ')
        return sentence.replace('-', ' ')

    def encode_parse(self, sentences: list, idf=False) -> np.array:
        """
        Takes a list of sentences, outputs a numpy array of sentence embeddings by computing the mean of words vector.
        Also use a parser to keep only important words (adjectives, verbs, common or proper nouns and interjections.
        If a word is unknown, ignore it, if a sentence is completely unknown, attribute a random vector as
        representation.
        """
        sentences_embedded = []
        for sent in sentences:
            sent = self.sentence_treated(sent)
            words_weights = []
            words_embedded = []
            for word in self.parser(sent):
                # Only keep important words, discard others
                if word.pos_ in ['ADJ', 'VERB', 'NOUN', 'PROPN', 'INTJ']:
                    str_word = str(word)
                    # Get embedding vector of each word of the sentence
                    try:
                        words_embedded.append(self.word2vec[str_word])
                        words_weights.append(
                            1 if idf is False else idf[str_word])  # Get weight of current word (idf or 1)
                    except KeyError:
                        # If word is unknown, ignore it.
                        if len(words_embedded) == len(words_weights) + 1:  # 2 different lists are used
                            words_weights.append(0)
                        continue
            # Average
            if len(words_embedded) > 0:
                sentences_embedded.append(np.average(words_embedded, weights=words_weights, axis=0))
            else:
                sentences_embedded.append(0.2 * np.random.rand(300) - 0.1)
        return np.array(sentences_embedded)
