import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import numpy as np
from sklearn.preprocessing import normalize

__authors__ = ['Paul Asquin', 'Benoit Laures', 'Ayush Rai']
__emails__ = ['paul.asquin@student.ecp.fr', 'benoit.laures@student.ecp.fr', 'ayush.rai2512@student-cs.fr']


class DialogueManager:
    def __init__(self):
        self.vect = TfidfVectorizer(analyzer='word',ngram_range=(1,1))

    def load(self,path):
        with open(path,'rb') as f:
            self.vect = pkl.load(f)


    def save(self,path):
        with open(path,'wb') as fout:
            pkl.dump(self.vect,fout)


    def train(self,data):
        self.vect.fit(data)


    def findBest(self,utterance,options):
        """
            finds the best utterance out of all those given in options
        :param utterance: a single string
        :param options: a sequence of strings
        :return: returns one of the strings of options
        """
        Xtext = [utterance] + options
        X = self.vect.transform(Xtext)
        X = normalize(X,axis=1,norm='l2')
        idx = np.argmax(X[0] * X[1:,:].T)

        return options[idx]


def loadData(path):
    """
        :param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)
        :return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue
            is a sequence of (utterance, answer, options)
    """
    with open(path) as f:
        descYou, descPartner = [], []
        dialogue = []
        for l in f:
            l=l.strip()
            lxx = l.split()
            idx = int(lxx[0])
            if idx == 1:
                if len(dialogue) != 0:
                    yield descYou,  descPartner, dialogue
                # reinit data structures
                descYou, descPartner = [], []
                dialogue = []

            if lxx[2] == 'persona:':
                # description of people involved
                if lxx[1] == 'your':
                    description = descYou
                elif lxx[1] == "partner's":
                    description = descPartner
                else:
                    assert 'Error, cannot recognize that persona ({}): {}'.format(lxx[1],l)
                description.append(lxx[3:])

            else:
                # the dialogue
                lxx = l.split('\t')
                utterance = ' '.join(lxx[0].split()[1:])
                answer = lxx[1]
                options = [o for o in lxx[-1].split('|')]
                dialogue.append( (idx, utterance, answer, options))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)
    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--gen', help='enters generative mode') # Can be a just a string

    opts = parser.parse_args()

    dm = DialogueManager()
    if opts.train and not opts.gen:
        print('Training Mode for Retrieval Based Model')
        text = []
        for _,_, dialogue in loadData(opts.text):
            for idx, _, _,options in dialogue:
                text.extend(options)
        dm.train(text)
        dm.save(opts.model)
    elif opts.test and not opts.gen:
        print('Test Mode for Retrieval Based Model')
        assert opts.test,opts.test
        dm.load(opts.model)
        for _,_, dialogue in loadData(opts.text):
            for idx, utterance, answer, options in dialogue:
                print(idx,dm.findBest(utterance,options))

    elif opts.train and opts.gen:
        print('Generative Model Training')

        # opts.gen can be a string

        #Load train dataset for Generative Model

        # Train the Generative Model

        # Save the Generative Model as opts.model

        # Loading the Test dataset using opts.text

        # Generate answers for each utterances in the test dataset

        pass

    elif opts.test and opts.gen:
        print('Generative Model Testing')
        #Load train dataset for Generative Model

        # Train the Generative Model

        # Save the Generative Model as opts.model

        # Loading the Test dataset using opts.text

        # Generate answers for each utterances in the test dataset

        pass