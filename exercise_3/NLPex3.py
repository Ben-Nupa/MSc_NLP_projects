import argparse
import warnings
warnings.filterwarnings("ignore")
from Retrieval_Dialog_Model import *

__authors__ = ['Paul Asquin', 'Benoit Laures', 'Ayush K. Rai']
__emails__ = ['paul.asquin@student.ecp.fr', 'benoit.laures@student.ecp.fr', 'ayush.rai2512@student-cs.fr']

NB_DIALOGUES_TRAIN = 1000
NB_DIALOGUES_VAL = 50
NUM_EPOCHS = 25
PATH_TO_PRETRAINED_GLOVE = 'resource/glove.6B.50d.txt'
EMBEDDING_DIM = 50
HIDDEN_LAYER_SIZE = 50
DROPOUT_PROB = 0.5

LEARNING_RATE = 1e-4
L2_PENALTY = 1e-4
COMPUTE_ACCURACY = False

PATH_WORD_TO_ID = './model/word_to_id.pkl'
PATH_ID_TO_WORD = './model/id_to_word.pkl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)
    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--gen', help='enters generative mode')

    opts = parser.parse_args()

    # Create a Retrieval Agent Dialog
    retrieval_dialog_agent = Retrieval_Dialog_Model()

    if opts.train and not opts.gen:
        # print('Training Mode for Retrieval Based Model')

        # Get Training Data and Create Training Variables
        id_to_vec, word_to_id, id_to_word, tr_my_personae, tr_other_personae, tr_line_indices, tr_utterances, tr_answers = retrieval_dialog_agent.creating_training_variables(
            opts.text, PATH_TO_PRETRAINED_GLOVE, embedding_dim=EMBEDDING_DIM, nb_dialogues=NB_DIALOGUES_TRAIN)

        # Saving the Extracted Training Data into Dataframes
        retrieval_dialog_agent.save_data_on_disk(word_to_id, id_to_word, tr_my_personae, tr_other_personae,
                                                 tr_line_indices, tr_utterances, tr_answers, True)

        # Create a Dual LSTM Model
        retrieval_model = retrieval_dialog_agent.creating_model(EMBEDDING_DIM, HIDDEN_LAYER_SIZE, DROPOUT_PROB,
                                                                id_to_vec)

        # Train the Model
        retrieval_model = retrieval_dialog_agent.train_model_df(retrieval_model, word_to_id, LEARNING_RATE, L2_PENALTY,
                                                                NUM_EPOCHS)

        # Save the Trained Model
        torch.save(retrieval_model, opts.model)

    elif opts.test and not opts.gen:

        # Perform Evaluation on the Validation and Test Dataset
        # print('Test Mode for Retrieval Based Model')
        retrieval_dialog_agent.test_model(opts.model, opts.text, PATH_WORD_TO_ID, PATH_ID_TO_WORD, NB_DIALOGUES_VAL,
                                          compute_accuracy=COMPUTE_ACCURACY)

        # print('Finished Testing')


    elif opts.train and opts.gen:
        print('Generative Model Training. We did not attempt this part')

        # opts.gen can be a string

        # Load train dataset for Generative Model

        # Train the Generative Model

        # Save the Generative Model as opts.model

        # Loading the Test dataset using opts.text

        # Generate answers for each utterances in the test dataset

        pass

    elif opts.test and opts.gen:
        print('Generative Model Testing. We did not attempt this part')
        # Load train dataset for Generative Model

        # Train the Generative Model

        # Save the Generative Model as opts.model

        # Loading the Test dataset using opts.text

        # Generate answers for each utterances in the test dataset

        pass
