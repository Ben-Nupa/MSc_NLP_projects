import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
import torch.nn.utils.rnn 
import datetime
import operator
from nltk.tokenize import word_tokenize
from typing import List
import pickle
import glob
import matplotlib.pyplot as plt
import sys
import pickle

import os
os.environ['CUDA_ENABLE_DEVICES'] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(device)

#-------------------Dual Encoder---------------------------------------------------------------------------------------------------------------------

class DualEncoder(nn.Module):
    """Dual LSTM encoder"""
     
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)     
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, context_tensor, response_tensor):
        
        context_last_hidden = self.encoder(context_tensor) #dimensions: (batch_size x hidden_size)
        response_last_hidden = self.encoder(response_tensor) #dimensions: (batch_size x hidden_size)
        
        #context = context_last_hidden.mm(self.M).cuda()
        context = context_last_hidden.mm(self.M) #dimensions: (batch_size x hidden_size)
        context = context.view(-1, 1, self.hidden_size) #dimensions: (batch_size x 1 x hidden_size)
        
        response = response_last_hidden.view(-1, self.hidden_size, 1) #dimensions: (batch_size x hidden_size x 1)
        
        #score = torch.bmm(context, response).view(-1, 1).cuda()
        score = torch.bmm(context, response).view(-1, 1) #dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)

        return score

#----------------------Encoder---------------------------------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    """LSTM encoder"""

    def __init__(self, emb_size, hidden_size, p_dropout, id_to_vec): 
    
            super(Encoder, self).__init__()
             
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            self.vocab_size = len(id_to_vec)
            self.p_dropout = p_dropout
       
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
            self.lstm = nn.LSTM(self.emb_size, self.hidden_size)
            self.dropout_layer = nn.Dropout(self.p_dropout) 

            self.init_weights(id_to_vec)
             
    def init_weights(self, id_to_vec):
        init.uniform_(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
        init.orthogonal_(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True
        
        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_size)
            
        for idx, vec in id_to_vec.items():
            embedding_weights[idx] = vec
        
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = True)
            
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        _, (last_hidden, _) = self.lstm(embeddings) #dimensions: (num_layers * num_directions x batch_size x hidden_size)
        last_hidden = self.dropout_layer(last_hidden[-1])#access last lstm layer, dimensions: (batch_size x hidden_size)

        return last_hidden


#----------------------------------------------------------------------------------------------------------------------------------------

class Retrieval_Dialog_Model:
    def __init__(self):
                
        if os.path.isdir("./model"):
            print('Model subdirectory already exists')
        else:
            os.makedirs('./model')
            print('Model subdirectory created')


        if os.path.isdir("./dataframes"):
            print('Dataframes subdirectory already exists')
        else:
            os.makedirs('./dataframes')
            print('Dataframes subdirectory created')




    def normalize(self,sentence: str) -> List[str]:
        """
        Normalize the given sentence by:
        - Converting to lower case
        - Transforming the English contractions to full words
        - Transforming composed words into 2 separated words
        - Tokenizing into words based on white space
        Inspired from : https://machinelearningmastery.com/clean-text-machine-learning-python/
        and https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
        """
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

        return word_tokenize(sentence)


    def extract_dataset_as_text(self, path: str, is_training_set: bool, nb_dialogues=-1) -> tuple:
        """
        Extract the dataset as text lists. If it is a training dataset, the first answer will be the correct one, others are
        distractors. To access an element of a dialogue i in one of the following list (e.g: output_list), do: output_list[i].

        Parameters
        ----------
        path : str
        Path to training file.
        is_training_set : bool
        Whether the dataset to extract is a training set (correct answer is known).
        nb_dialogues : int
        Number of dialogues to extract. Set -1 for all.

        Returns
        -------
        out : tuple
        word_to_id : Dict[str, int]
            Vocabulary mapping each word to an unique id. Only returned if it's a training set.
        my_personae : List[List[List[str]]]
            My personae of each dialogue.
        other_personae : List[List[List[str]]]
            Other personae of each dialogue.
        line_indices : List[List[int]]
            Indices of lines, except those describing the persona, of each dialogue.
        utterances : List[List[List[str]]]
            Utterances (question-like) of each dialogue.
        answers : List[List[List[List[str]]]]
            Answers of each utterance of each dialogue. The correct one (for a training set) for a dialogue i and an
            utterance j is answers[i][j][0], the others answers[i][j][k] for k>0 are wrong answers.
        """
        def get_tokens_from_sentence(sentence: str):
            """Normalizes, extracts the tokens from the given sentence and adds them to the vocabulary."""
            tokens = self.normalize(sentence)
            #id_to_word = {}
            if is_training_set:
                for element in tokens:
                    if element not in word_to_id:
                        id_to_word[len(word_to_id)] = element
                        word_to_id[element] = len(word_to_id)
            return tokens
                
        word_to_id = {}
        id_to_word = {}
        my_personae = []
        other_personae = []
        line_indices = []
        utterances = []
        answers = []
        idx_dialogue = 0

        with open(path, 'r') as file:
            for line in file:
                words = line.split()
                idx_line = int(words[0])
                if idx_line == 1:
                    idx_dialogue += 1
                    if idx_dialogue == nb_dialogues + 1:
                        break

                # Get my persona
                if words[1] + ' ' + words[2] == 'your persona:':
                    if len(my_personae) != idx_dialogue:
                        my_personae.append([])
                    my_personae[-1].append(get_tokens_from_sentence(' '.join(str(word) for word in words[3:])))

                # Get other persona
                elif words[1] + ' ' + words[2] == "partner's persona:":
                    if len(other_personae) != idx_dialogue:
                        other_personae.append([])
                    other_personae[-1].append(get_tokens_from_sentence(' '.join(str(word) for word in words[3:])))

                # Get dialogue
                else:
                    if len(utterances) != idx_dialogue:
                        line_indices.append([])
                        utterances.append([])
                        answers.append([])

                    line_indices[-1].append(idx_line)
                    exchange = line[len(str(idx_line)) + 1:].split('\t')
                    utterances[-1].append(get_tokens_from_sentence(exchange[0]))
                    # Training set: answer is known
                    if is_training_set:

                        answers[-1].append([get_tokens_from_sentence(exchange[1])])  # Correct answers
                        correct_answer = get_tokens_from_sentence(exchange[1]) # Newly Added 

                        for statement in exchange[2:]:  # Wrong answers
                     
                            if statement == '':
                                continue
                            for distractor in statement.split('|'):
                                splitted_distractor = get_tokens_from_sentence(distractor)

                                if splitted_distractor == correct_answer:
                            	    continue

                                answers[-1][-1].append(get_tokens_from_sentence(distractor))
                            
                    # Testing set: answer is unknown
                    else:
                        answers[-1].append([])
                        for statement in exchange[1:]:
                            if statement == '':
                                continue
                            for distractor in statement.split('|'):
                                answers[-1][-1].append(get_tokens_from_sentence(distractor))
                            
        print('Loaded ' + str(len(line_indices)) + ' dialogues')
        if is_training_set:
            return word_to_id, id_to_word ,my_personae, other_personae, line_indices, utterances, answers
        else:
            return my_personae, other_personae, line_indices, utterances, answers

        #----------------------------------------------------------------------------------------------------------------

    def create_id_to_vec(self,word_to_id: dict, path_to_glove_weights: str) -> dict:
        """
        Extracts the embedding weights for each word in the vocabulary and maps each word ids to its weight in a dictionary.
    
        Parameters
        ----------
        path_to_glove_weights : str
        Path to the file containing the embedding weights.
        word_to_id : Dict[str, int]
        Vocabulary mapping each word to an unique id.

        Returns
        -------
        id_to_vec : Dict[int, np.ndarray]
        Map of each word id to its embedding form.
        """
        with open(path_to_glove_weights, 'r', encoding='utf-8') as glovefile:
            lines = glovefile.readlines()

        id_to_vec = {}
        vector = None
    
        for line in lines:
            word = line.split()[0]
            vector = np.array(line.split()[1:], dtype='float32')
        
            if word in word_to_id:
                id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
            
        for word, id in word_to_id.items(): 
            if word_to_id[word] not in id_to_vec:
                v = np.zeros(*vector.shape, dtype='float32')
                v[:] = np.random.randn(*v.shape)*0.01
                id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
                
        return id_to_vec
        #---------------------------------------------------------------------------------------------------------------------------------

    def creating_training_variables(self,path_to_training_set, path_to_glove_weights, embedding_dim=50, nb_dialogues=-1):
        print(str(datetime.datetime.now()).split('.')[0], "Creating variables for training...")
    
        word_to_id, id_to_word ,my_personae, other_personae, line_indices, utterances, answers = self.extract_dataset_as_text(path_to_training_set, True, nb_dialogues)
        id_to_vec = self.create_id_to_vec(word_to_id, path_to_glove_weights)
        # Unknown words
        v = np.zeros(embedding_dim, dtype='float32')
        v[:] = np.random.randn(*v.shape)*0.01
        id_to_vec[-1] = torch.FloatTensor(torch.from_numpy(v))

        print(str(datetime.datetime.now()).split('.')[0], "Variables created.\n")
        return id_to_vec, word_to_id, id_to_word ,my_personae, other_personae, line_indices, utterances, answers

        #---------------------------------------------------------------------------------------------------------------------------------

    def creating_validation_variables(self,path_to_validation_set, nb_dialogues=-1):
        print(str(datetime.datetime.now()).split('.')[0], "Creating variables for validations...")
    
        _, _ ,my_personae, other_personae, line_indices, utterances, answers = self.extract_dataset_as_text(path_to_validation_set, True, nb_dialogues)

        print(str(datetime.datetime.now()).split('.')[0], "Variables created.\n")
        return my_personae, other_personae, line_indices, utterances, answers

        #-----------------------------------------------------------------------------------------------------------------------------------

    def creating_model(self, emb_size, hidden_size, p_dropout, id_to_vec):

        print(str(datetime.datetime.now()).split('.')[0], "Calling model...")

        encoder = Encoder(emb_size, hidden_size, p_dropout, id_to_vec)

        dual_encoder = DualEncoder(encoder)

        print(str(datetime.datetime.now()).split('.')[0], "Model created.\n")
        print(dual_encoder)
    
        return dual_encoder.to(device)

        #------------------------------------------------------------------------------------------------------------------------------------

    def get_word_id(self, word_to_id: dict, token: str) -> int:
        """Retrieves the ID of the word if known, else returns -1 (ID for unknown words)."""
        try:
            id_word = word_to_id[token]
        except KeyError:
            id_word = 0
        return id_word

        #-----------------------------------------------------------------------------------------------------------------------------------

    def save_data_on_disk(self, word_to_id, id_to_word ,my_personae, other_personae, line_indices, utterances, answers, is_training, max_context_len=50, max_size_df=10000):
        dataframe_name = 'validation_df'
        if is_training:
            with open('model/word_to_id' + '.pkl', 'wb') as dict_file:
                pickle.dump(word_to_id, dict_file)

            with open('model/id_to_word' + '.pkl', 'wb') as dict_file2:
                pickle.dump(id_to_word, dict_file2)
            dataframe_name = 'training_df'
        
        dataframe_to_save = pd.DataFrame(columns=['context', 'response', 'label', 'idx_line','text'])
        idx_dataframe = 0
        for idx_dialogue in range(len(line_indices)):
            if len(dataframe_to_save) >= max_size_df:
                dataframe_to_save.to_csv('dataframes/{0}{1}.csv'.format(dataframe_name, idx_dataframe), index=False)
                idx_dataframe += 1
            
            context_ids = []
            # Add my persona in context
            for sentence in my_personae[idx_dialogue]:
                for token in sentence:
                    context_ids.append(self.get_word_id(word_to_id, token))

            # Add other persona in context
            for sentence in other_personae[idx_dialogue]:
                for token in sentence:
                    context_ids.append(self.get_word_id(word_to_id, token))

            # Add utterances, create responses and labels
            for idx_utterance in range(len(utterances[idx_dialogue])):
                if idx_utterance != 0:
                    # Add previous correct answer in context
                    for token in answers[idx_dialogue][idx_utterance - 1][0]:
                        context_ids.append(self.get_word_id(word_to_id, token))

                # Add utterances in context
                for token in utterances[idx_dialogue][idx_utterance]:
                    context_ids.append(self.get_word_id(word_to_id, token))

                # Get response and label
                for idx_answer in range(len(answers[idx_dialogue][idx_utterance])):
                    response_ids = []
                    
                    for token in answers[idx_dialogue][idx_utterance][idx_answer]:
                        response_ids.append(self.get_word_id(word_to_id, token))


                    if idx_answer == 0:
                        label = 1
                    else:
                        label = 0
                    
                    if len(context_ids) > max_context_len:
                        context_ids = context_ids[-max_context_len:]
                    if len(response_ids) > max_context_len:
                        response_ids = response_ids[-max_context_len:]

                    dataframe_to_save.loc[len(dataframe_to_save)] = [0, 0, label, line_indices[idx_dialogue][idx_utterance],0]
                    dataframe_to_save['context'][len(dataframe_to_save) - 1] = context_ids
                    dataframe_to_save['response'][len(dataframe_to_save) - 1] = response_ids
                    dataframe_to_save['text'][len(dataframe_to_save) - 1] = ' '.join(answers[idx_dialogue][idx_utterance][idx_answer])
    
        dataframe_to_save.to_csv('dataframes/{0}{1}.csv'.format(dataframe_name, idx_dataframe), index=False)

        #-----------------------------------------------------------------------------------------------------------------------------------------------------

    def plot_loss(self, train_loss_data,epoch_list):
    
        plt.plot(epoch_list, train_loss_data, linewidth=2.0,linestyle='-',color='darkcyan',label='Train')
        plt.title('Plot '+'for Training Loss')
        plt.xlabel('# of Epochs')
        plt.ylabel("Training Loss")
        plt.legend(loc='best')
        plt.grid(axis='both')
        plt.savefig('./training_loss.png')
        #-------------------------------------------------------------------------------------------------------------------------------------------------------

    def train_model_df(self, dual_encoder, word_to_id, learning_rate=1e-4, l2_penalty=1e-4, nb_epochs=25):
        """Training with dataframe"""
        print(str(datetime.datetime.now()).split('.')[0], "Starting training...\n")

        optimizer = torch.optim.Adam(dual_encoder.parameters(), lr = learning_rate, weight_decay = l2_penalty)
        loss_func = torch.nn.BCEWithLogitsLoss()

        training_loss_list = []
        epochs_list = []
          
        for epoch in range(nb_epochs):
            print("Epoch : ", epoch, " / ", nb_epochs)
            sum_loss_training = 0
            nb_iter_tr = 0
            sum_loss_validation = 0
            nb_iter_val = 0

            # First: use training set
            dual_encoder.train()
            for training_df_name in glob.glob('dataframes/training_df*'):
                training_df = pd.read_csv(training_df_name).sample(frac=1)  # Shuffle
            
                for idx, row in training_df.iterrows():
            
                    context_ids = list(map(int, row['context'][1:-1].split(', ')))
                    response_ids = list(map(int, row['response'][1:-1].split(', ')))
                    label = np.array(row['label']).astype(np.float32)

                    context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1), requires_grad = False).cuda()
                    response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1), requires_grad = False).cuda()   
                    label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1, 1))), requires_grad = False).cuda()
                    #print('Label: ',label.data)

                    # Predict
                    score = dual_encoder(context, response)
                    loss = loss_func(score, label)

                    # Train
                    nb_iter_tr += 1
                    sum_loss_training += loss.data.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
            # Second: use validation set
            #dual_encoder.eval()

            #for validation_df_name in glob.glob('data/validation_df*'):
            #    validation_df = pd.read_csv(validation_df_name).sample(frac=1)  # Shuffle

            #    for idx, row in validation_df.iterrows():

            #        context_ids = list(map(int, row['context'][1:-1].split(', ')))
            #        response_ids = list(map(int, row['response'][1:-1].split(', ')))
            #        label = np.array(row['label']).astype(np.float32)

            #        context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1), requires_grad = False).cuda()
            #        response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1), requires_grad = False) .cuda()   
            #        label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1, 1))), requires_grad = False).cuda()

    #                # Predict
            #        score = dual_encoder(context, response)
            #        loss = loss_func(score, label)
        #            nb_iter_val += 1
        #            sum_loss_validation += loss.data.item()
        
        
            #print('Num Iter: ', nb_iter_tr)
            training_loss = sum_loss_training/nb_iter_tr
            print('Training loss =', training_loss)
            training_loss_list.append(training_loss)
            epochs_list.append(epoch)
        
            #print('Validation loss =', sum_loss_validation / nb_iter_val)
                
        print(str(datetime.datetime.now()).split('.')[0], "Training and validation epochs finished.")
        self.plot_loss(training_loss_list,epochs_list)
        return dual_encoder

        #---------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_word_from_id(self, id_to_word,word_id_list):

        answer = []

        for id in word_id_list:
            try:
                answer.append(id_to_word[id])
            except KeyError:
                answer.append('UNK')

        #print(answer)

        answer = ' '.join(answer)

        #print(answer)

        return answer

        #-----------------------------------------------------------------------------------------------------------------------------------------------------------

    def test_model(self, retrieval_dialog_model_path,test_data_path,path_word_to_id,path_id_to_word,NB_DIALOGUES_VAL,compute_accuracy=False):
        """Training with dataframe"""
        print(str(datetime.datetime.now()).split('.')[0], "Starting Testing...\n")

        retrieval_dialog_model = torch.load(retrieval_dialog_model_path)
        retrieval_dialog_model.to(device)

        print('Retrieval Model Loaded Successful')

        with open(path_word_to_id, "rb") as dict_file:
            word_to_id = pickle.load(dict_file)

        #print(word_to_id)

        print('Loaded Word to id dict')

        with open(path_id_to_word, "rb") as dict_file2:
            id_to_word = pickle.load(dict_file2)

        print('Loaded id to word dict')

        #print(id_to_word)


        val_my_personae, val_other_personae, val_line_indices, val_utterances, val_answers = self.creating_validation_variables(test_data_path, NB_DIALOGUES_VAL)

        self.save_data_on_disk(word_to_id, id_to_word ,val_my_personae, val_other_personae, val_line_indices, val_utterances, val_answers, False)

        retrieval_dialog_model.eval()

        total = 0
        correct = 0

        for validation_df_name in glob.glob('dataframes/validation_df*'):
            validation_df = pd.read_csv(validation_df_name)  # Shuffle

            context_list = list(dict.fromkeys(list(validation_df['context'])))

            #print(len(context_list))

            for cntxt in context_list:

                val_temp_df = validation_df.loc[validation_df['context'] == cntxt]

                #Store the Groundtruth and Match with the Best Retrieved Answer
                groundtruth_answer = val_temp_df.iloc[0]['text']

                #print(type(groundtruth_answer))
                #print(groundtruth_answer)

                val_temp_df = val_temp_df.drop(val_temp_df.index[0])
                score_temp_list = []

                total += 1

                for idx, row in val_temp_df.iterrows():

                    context_ids = list(map(int, row['context'][1:-1].split(', ')))
                    response_ids = list(map(int, row['response'][1:-1].split(', ')))
                    label = np.array(row['label']).astype(np.float32)

                    context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1), requires_grad = False).cuda()
                    response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1), requires_grad = False).cuda()   
                    label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1, 1))), requires_grad = False).cuda()
                    
                    # Predict
                    score = retrieval_dialog_model(context, response)
                    #loss = loss_func(score, label)
                    #_, predicted = score.max(1)
                    score = score.cpu().detach().numpy()[0][0]
                    score_temp_list.append(score)

                val_temp_df['Predicted_Score'] = score_temp_list


                #print(val_temp_df)
                #print(val_temp_df['Predicted_Score'])
                #print(val_temp_df['Predicted_Score'].idxmax())

                best_answer_df = val_temp_df.loc[val_temp_df['Predicted_Score'].idxmax()]

                #best_answer = self.get_word_from_id(id_to_word,best_answer_df)

                selected_best_answer = best_answer_df['text']

                # Checking whether selected answer is same as the groundtruth
                if(selected_best_answer==groundtruth_answer):
                	correct += 1

                print(best_answer_df['idx_line'],selected_best_answer)

        if compute_accuracy:
        	accuracy = (100*correct)/total
        	print('\nRetrieval Accuracy :', accuracy)

        #print('\nBest Retrieved Answers')
        #----------------------------------------------------------------------------------------------------------------------------------------------------------


    def load(self, path):
        pass


        #-----------------------------------------------------------------------------------------------------------------------------------------------------------

    def save(self, path):
        pass


        #------------------------------------------------------------------------------------------------------------------------------------------------------------

    def train(self, data):
        pass


        #-------------------------------------------------------------------------------------------------------------------------------------------------------------

    def findBest(self, utterance, options):
        pass



        #----------------------------------------------------------------------------------------------------------------------------------------------------------------