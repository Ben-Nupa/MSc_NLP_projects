import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from word2vec import PreTrainedWord2Vec


class Classifier:
    """The Classifier (Word2Vec embedding + Logistic Regression"""

    def __init__(self, path_to_weights='../resources/crawl-300d-200k.vec'):
        """
        Initialize the classifier.
        @:param path_to_weights : path to the file containing the weights for the pretrained word2vec model.
        """
        # Data
        self.col_names = ['Polarity', 'Aspect_Category', 'Specific_Target_Aspect_Term', 'Character_Offset', 'Sentence']
        self.polarity_encoder = LabelEncoder()
        self.categories_encoder = LabelEncoder()
        # Model
        self.w2v = PreTrainedWord2Vec(path_to_weights, 150000)
        self.clf = LogisticRegression(C=1, solver='liblinear', multi_class='ovr')

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        # Load data
        train_df = pd.read_csv(trainfile, sep='\t', names=self.col_names)

        # Load training matrix as word2vec
        X_tr = self.w2v.encode_parse(train_df.Sentence, False)

        # Add categories information
        train_categories_integer = self.categories_encoder.fit_transform(train_df.Aspect_Category)
        train_categories_dummy = to_categorical(train_categories_integer)
        X_tr = np.hstack((X_tr, train_categories_dummy))

        # Get training labels
        y_tr = self.polarity_encoder.fit_transform(train_df.Polarity)
        self.clf.fit(X_tr, y_tr)

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        # Load data
        test_df = pd.read_csv(datafile, sep='\t', names=self.col_names)

        # Load training matrix as word2vec
        X_te = self.w2v.encode_parse(test_df.Sentence, False)

        # Add categories information
        test_categories_integer = self.categories_encoder.transform(test_df.Aspect_Category)
        test_categories_dummy = to_categorical(test_categories_integer)
        X_te = np.hstack((X_te, test_categories_dummy))

        return list(self.polarity_encoder.inverse_transform(self.clf.predict(X_te)))
