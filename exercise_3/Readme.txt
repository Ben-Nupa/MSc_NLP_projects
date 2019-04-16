# To run the Code

1) Download 50 dimensional Pretrained Glove Embeddings from https://www.kaggle.com/watts2/glove6b50dtxt#glove.6B.50d.txt
2) Create the resource folder in the same directory as the NLPex3.py and put these downloaded embedding inside it


---------------------------
Retrieval Based Model:
----------------------------

For Training
python NLPex3.py --train --model ./model/retrieval_model.pth --text ./data/train_both_original.txt

For Validation/Testing
python NLPex3.py --test --model ./model/retrieval_model.pth --text ./data/valid_both_original.txt

---------------------
Generative Model:
--------------------

For training
python NLPex3.py --train --model ./model/generative_model.pth --text ./data/train_both_original.txt --gen Geneative_Mode

For Testing
python NLPex3.py --test --model ./model/generative_model.pth --text ./data/valid_both_original.txt --gen Geneative_Mode

