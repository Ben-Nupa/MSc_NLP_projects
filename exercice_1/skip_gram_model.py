import numpy as np
import matplotlib.pyplot as plt


class SkipGram:
    """
    Skip-Gram model

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (total number of unique words).
    word_frequencies : ndarray
         Frequency of each word in the same order as words ids.
    embed_dim : int
        Dimension of embedding space.
    Attributes
    ----------
    TODO
    """
    def __init__(self, vocab_size: int, word_frequencies: np.ndarray, embed_dim=100):
        # Dimensions
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.neg_sampling_distrib = self.compute_negative_sampling_distribution(
            word_frequencies)  # For negative sampling

        # Network variables
        self.w1 = np.array([])  # Shape (vocab_size, embed_dim)
        self.h = np.array([])  # Shape (-1, embed_dim)
        self.w2 = np.array([])  # Shape (embed_dim, vocab_size)
        self.score = np.array([])  # Shape (-1, vocab_size)
        self.probabilities = np.array([])  # Shape (-1, vocab_size)

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes the weights of the layers with the same method as Word2Vec."""
        self.w1 = np.random.uniform(-1 / (2 * self.embed_dim), 1 / (2 * self.embed_dim),
                                    size=(self.vocab_size, self.embed_dim))
        self.w2 = np.zeros((self.embed_dim, self.vocab_size))

    def softmax(self, y_ids=None, neg_sampling_size=5):
        """
        Computes the probability vector for each class. By default, doesn't use negative sampling: only uses it if
        y_ids is not None and if training time.

        Parameters
        ----------
        y_ids : list
            IDs of output words.
        neg_sampling_size : int
            Size of negative sampling. Advised is 5 - 20 for small datasets and 2 - 5 for large datasets.
        """
        if y_ids is None:
            exp_s = np.exp(self.score)
            self.probabilities = exp_s / np.sum(exp_s, axis=1).reshape(-1, 1)

        else:  # Do negative sampling
            neg_sampling_idx = np.random.choice(np.arange(len(self.neg_sampling_distrib)), size=neg_sampling_size,
                                                p=self.neg_sampling_distrib)
            # Make sure we sample negative examples
            while not set(neg_sampling_idx).intersection(set(y_ids)) == set():
                neg_sampling_idx = np.random.choice(np.arange(len(self.neg_sampling_distrib)), size=neg_sampling_size,
                                                    p=self.neg_sampling_distrib)

            neg_sampling_idx = np.append(neg_sampling_idx, y_ids)
            exp_s = np.exp(self.score[:, neg_sampling_idx])
            self.probabilities = np.zeros(self.score.shape)
            self.probabilities[:, neg_sampling_idx] = exp_s / np.sum(exp_s, axis=1).reshape(-1, 1)

    def forward_pass(self, x, y_ids=None, neg_sampling_size=5):
        self.h = x.dot(self.w1)
        self.score = self.h.dot(self.w2)
        self.softmax(y_ids, neg_sampling_size)

    def compute_loss(self, x, y, y_ids=None):
        self.forward_pass(x, y_ids)
        loss = -np.log(np.sum(y * self.probabilities, axis=1))
        # loss = -np.log(np.sum(y.multiply(self.probabilities), axis=1))
        return np.sum(loss) / x.shape[0]

    def compute_gradients(self, x, y):
        """Computes the analytical gradients of J w.r.t b1, W1, b2 and W2 with L2-regularization for the given parameters.
        @:return ndarray grad_b1 (m, 1), ndarray grad_W1 (m, d), ndarray grad_b2 (K, 1), ndarray grad_W2 (K, m)"""
        # x.shape[0] is the batch size
        g = -(y - self.probabilities) / x.shape[0]  # Shape (-1, vocab_size)
        grad_w2 = self.h.T.dot(g)  # Shape (embed_dim, vocab_size)

        g = g.dot(self.w2.T)  # Shape (-1, embed_dim)
        grad_w1 = x.T.dot(g)  # Shape (vocab_size, embed_dim)

        return grad_w1, grad_w2

    def backward_pass(self, x, y, learning_rate):
        """Performs a backward pass and actualizes the value of all gradients."""
        grad_w1, grad_w2 = self.compute_gradients(x, y)
        self.w1 -= learning_rate * grad_w1
        self.w2 -= learning_rate * grad_w2

    @staticmethod
    def compute_negative_sampling_distribution(word_frequencies: np.ndarray, exponent=0.75):
        word_frequencies = word_frequencies ** exponent
        return word_frequencies / np.sum(word_frequencies)

    def train(self, x, y, y_ids=None, n_epochs=10, batch_size=16, neg_sampling_size=5, learning_rate=1e-2,
              decay_factor=1):
        self.initialize_weights()
        loss_training_set = []
        for idx_epoch in range(n_epochs):
            print("Performing epoch " + str(idx_epoch) + "/" + str(n_epochs))
            # Batch indices
            batch_indices = list(range(0, x.shape[0], batch_size))
            np.random.shuffle(batch_indices)

            # Train by batch
            for j in batch_indices:
                x_batch = x[j:j + batch_size, :]
                y_batch = y[j:j + batch_size, :]

                if y_ids is not None:
                    y_ids_batch = y_ids[j:j + batch_size]
                else:
                    y_ids_batch = None

                self.forward_pass(x_batch, y_ids_batch, neg_sampling_size)
                self.backward_pass(x_batch, y_batch, learning_rate)

            # Decay learning rate regularly
            if idx_epoch % 2 == 0:
                learning_rate *= decay_factor
            # Compute loss
            loss_training_set.append(self.compute_loss(x, y, y_ids))

        # Plot
        fig = plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(np.arange(1, n_epochs + 1), loss_training_set, label="Training set")
        plt.legend()

    def predict(self, x):
        """Predicts the class of the given data as one-hot encoded vectors.
        @:return ndarray predictions (K, N)"""
        self.forward_pass(x)
        labels = np.argmax(self.probabilities, axis=0)
        y_pred = np.zeros((x.shape[0], self.vocab_size))
        for i in range(x.shape[0]):
            y_pred[i, labels] = 1
        return y_pred

    def embed(self, x):
        return x.dot(self.w1)

    def save(self, path):
        raise NotImplementedError('implement it!')

    def similarity(self, word1, word2):
        """
        Computes the cosine similarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        word1_embedded = self.embed(word1)
        word2_embedded = self.embed(word2)
        return word1_embedded.dot(word2_embedded) / (np.linalg.norm(word1_embedded) * np.linalg.norm(word2_embedded))

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

    def compute_grads_num_slow(self, x, y, y_ids, step):
        """Computes the numerical gradients of J w.r.t b1, W1, b2 and W2 with L2-regularization for the given parameters,
        using the centered difference formula to compute the gradients.
        @:return ndarray grad_b1 (m, 1), ndarray grad_W1 (m, d), ndarray grad_b2 (K, 1), ndarray grad_W2 (K, m)"""
        grad_w1 = np.zeros(self.w1.shape)
        grad_w2 = np.zeros(self.w2.shape)
        original_w1 = np.copy(self.w1)
        original_w2 = np.copy(self.w2)
        for i in range(self.embed_dim):
            for j in range(self.vocab_size):
                # Compute numerical gradient for w1
                self.w1[j, i] -= step
                c_minus = self.compute_loss(x, y, y_ids)
                self.w1 = np.copy(original_w1)
                self.w1[j, i] += step
                c_plus = self.compute_loss(x, y, y_ids)
                grad_w1[j, i] = (c_plus - c_minus) / (2 * step)
                self.w1 = np.copy(original_w1)

                # Compute numerical gradient for w2
                self.w2[i, j] -= step
                c_minus = self.compute_loss(x, y, y_ids)
                self.w2 = np.copy(original_w2)
                self.w2[i, j] += step
                c_plus = self.compute_loss(x, y, y_ids)
                grad_w2[i, j] = (c_plus - c_minus) / (2 * step)
                self.w2 = np.copy(original_w2)

        return grad_w1, grad_w2

    def compare_gradients(self, x, y, y_ids=None, step=1e-5):
        """Compares the gradients obtained with the analytical and numerical methods, computing the relative error as
        explained in the Additional material for lecture 3 from Standford's course Convolutional Neural Networks for
        Visual Recognition
        @:return list of float relative errors for b1, W1, b2, W2 gradients"""

        def compute_relative_error(analytical_gradient, numerical_gradient):
            max_array = np.maximum(np.abs(analytical_gradient), np.abs(numerical_gradient))
            max_array[np.where(max_array == 0)] = 1  # To make sure we don't divide by 0
            relative_error = np.abs(analytical_gradient - numerical_gradient) / max_array
            return relative_error

        self.forward_pass(x, y_ids)
        analytical_grad_w1, analytical_grad_w2 = self.compute_gradients(x, y)
        print('Computed analytical gradient.')
        numerical_grad_w1, numerical_grad_w2 = self.compute_grads_num_slow(x, y, y_ids, step)
        print('Computed numerical gradient.')

        relative_error_w1 = compute_relative_error(analytical_grad_w1, numerical_grad_w1)
        relative_error_w2 = compute_relative_error(analytical_grad_w2, numerical_grad_w2)
        print('Error gradient w1 =', np.mean(relative_error_w1))
        print('Error gradient w2 =', np.mean(relative_error_w2))
        return relative_error_w1, relative_error_w2
