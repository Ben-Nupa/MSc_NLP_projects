Natural Language Processing Course - Assignment 2 - Aspect Based Sentiment Analysis

1) Authors : Benoit Laures
          Ayush K. Rai
          Paul Asquin

2) In order to run the code:

    a) Create a "resources" folder (in the same directory as the src and data folder)

    b) Download the 300 dimensional pretrained FastText Word Embeddings (trained on Common Crawl Dataset) and put it in the "resources" folder :  https://centralesupelec-my.sharepoint.com/personal/benoit_laures_student_ecp_fr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fbenoit_laures_student_ecp_fr%2FDocuments%2Fcrawl-300d-200k%2Evec&parent=%2Fpersonal%2Fbenoit_laures_student_ecp_fr%2FDocuments&cid=c8373839-d7c5-4a18-84c1-2859ff2b7eb4

    python3 tester.py

3) Description of the Final Pipeline

   In order to attack the problem of Aspect Based Sentiment Analysis, we use the following strategy

   a) We use the Parser from the Spacy Library to extract adjectives, verbs, common or proper nouns and interjections from the sentences, which we call sentiment specific words. The intuition behind this is that these figures of speech capture information about the corresponding Aspect category.
   b) The next step in our pipeline involves transformation of the extracted sentiment specific words into the feature vector by using 300 dimensional pretrained FastText Embeddings (we load only 150 000 words to avoid out of memory issues). For this, we represent each word of the sentence by its embedding form and we compute the sentence representation as the mean of the embedding form of the words.
   c) We take into account the 12 aspects categories. To do so, we transform these categories into dummy features and concatenate them to the embedding features. Hence, a sentence is represented by a vector of dimension 312.
   d) Finally we apply multinomial logistic regression machine learning model for classification. This was a difficult choice for us and we performed a lot of experiments. But highly complex models (like LTSM based models, Conv1D based model, fully connected model, SVM with RBF kernel, ensemble methods, etc...) did not outperform the multinomial logistic regression model. Therefore following the Occam's Razor principle, we choose the Multinomail Logistic Regression model.


4) Accuracy on the Dev Set : 0.8138

Additional Note:

Other Models we tried which didn't perform well

1) Using Spacy Parser to extract the adjectives and verbs followed by feature extraction using Bag of Words Model and finally applying fully connected neural network for classification. We achieved the accuracy of around 0.773 using this model.

2) Another approach we tried is to extract sentence level features using pretrained Glove Embeddings (with dimension varying from 50 to 300) and then apply multiple machine/deep learning models like Logistic Regression, Conv1D, Random Forest foraspect based sentiment analysis. However we achieved a highest accuracy of 0.78 by using these models.
