--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Retrieval Based Dialog Agent:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Code is successfully tested on Python >=3.5, PyTorch >=1.0

# To run the Code

1) Download 50 dimensional Pretrained Glove Embeddings (trained on Common Crawl Dataset) from https://www.kaggle.com/watts2/glove6b50dtxt#glove.6B.50d.txt
2) Create the `resource` folder in the same directory as the NLPex3.py and put these downloaded embedding inside it (or put it in any folder and modify the variable `PATH_TO_PRETRAINED_GLOVE`.

For Training
    python NLPex3.py --train --model <path_to_model> --text <path_to_training_data>

    For example:
        python NLPex3.py --train --model model/retrieval_model.pth --text data/train_both_original.txt


For Validation/Testing
    python NLPex3.py --test --model <path_to_model> --text <path_to_validation_data>

    For example:
        python NLPex3.py --test --model model/retrieval_model.pth --text data/valid_both_original.txt


By default, the model is trained on 1000 dialogues (about 320,000 training examples) and tested on 50 dialogues (about 16,000 examples). If it takes too much time, you can diminish this number with the variables `NB_DIALOGUES_TRAIN` and `NB_DIALOGUES_VAL` located in the file `NLPex3.py`.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Generative Dialog Agent:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For Training
python NLPex3.py --train --model ./model/generative_model.pth --text ./data/train_both_original.txt --gen Geneative_Mode

We did not attempt this part.

For Validation/Testing
python NLPex3.py --test --model ./model/generative_model.pth --text ./data/valid_both_original.txt --gen Geneative_Mode

We did not attempt this part.

