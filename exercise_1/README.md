# Exercise 1: Skip-gram with negative-sampling from scratch

## Run
To train:
```bash
python3 skipGram.py --text <relative_path_to_folder> --model <model_name.pkl>
```
To test:
```bash
python3 skipGram.py --text <relative_path_to_EN-SIMLEX-999.txt> --model <model_name.pkl> --test
```

For example if the data is in a repository `data`, `cd exercice_1`.

To train, type:
```bash
python3 skipGram.py --text data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled --model model.pkl
```
To test, type:
```bash
python3 skipGram.py --text data/EN-SIMLEX-999.txt --model model.pkl --test
```

## Default parameters
Here as the default paramters you will find when launching the training:
- Number of lines = 100000
- Size of context window = 3
- Number of epochs = 500
- Embedded size = 300
- Number of words sampled for negative sampling = 5
- Size of 1 batch = 256
- Initial learning rate value = 5e-3
- Decay factor of the learning rate = 0.99
- Interval of epochs between 2 decays = 5

All these parameters are defined as global variables and can be changed in the file `skipGram.py`.

## Pipeline
One-hot encoding of data:
- Read file line by line until the desired number of lines is achieved
- Data filtering : set to lowercase, remove all non alphanumerical characters except spaces and ', transform ' to spaces
- Map each word to an ID and vice versa in a dictionnary
- Generate the datasets in 2 different forms: list of indices of words and one-hot sparse encoded matrices. The i-th element of the X list corresponds to a center word and the i-th element of the Y list corresponds to a context word of this center word. For instance if the center word '20' has the contex words (let's say window size of 3) '52', '38', '11', '0', '124', '89'; we'll have `X = [..., 20, 20, 20, 20, 20, 20, ...]` and `Y = [..., 52, 38, 11, 0, 124, 89, ...]`.

For the training:
- Weights initialization: Uniform law over <img src="https://latex.codecogs.com/gif.latex?[\frac{-1}{2n_{embed}},&space;\frac{1}{2n_{embed}}]" title="[\frac{-1}{2n_{embed}}, \frac{1}{2n_{embed}}]" /> for the encoding W1 matrix and 0-matrix for the decoding W2 matrix.
- Training with batch (a center word can be in several batches) with negative sampling for the softmax approximation, saturation on exponential to avoid overflow, gradient clipping to prevent it from exploding and SGD algorithm for the update.
- By default, the learning rate is decayed (default factor is 0.99) after each epoch and the total loss is computed: the learning curve is plot at the end of the training.

To compute the similarity, the cosine similarity is used. If the similarity score given is `-1`, this means we didn't had this word in our vocabulary (unknown words handling).

## Thought process

### Getting a working model
The skip-gram model is simply a fully-connected neural network with one hidden layer without activation function, thus we built a class for this general kind of networks. A focus was made to compute the analytical gradients so as to make computation as effective as possible (numpy matrices-based operations and no 'for loop'): to check that the gradients were correct, we compared them to the numerical gradient scomputed with the central differencing scheme and by looking at the relative error. The following operations are applied:

<img src="https://latex.codecogs.com/gif.latex?g=(p&space;-&space;y)/B" title="g=(p - y)/B" />

<img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;W_2&space;=&space;h^\top&space;g" title="\bigtriangledown W_2 = h^\top g" />

<img src="https://latex.codecogs.com/gif.latex?g=gW_2^\top" title="g=gW_2^\top" />

<img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;W_1&space;=&space;x^\top&space;g" title="\bigtriangledown W_1 = x^\top g" />

where <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> is the one-hot encoded output, <img src="https://latex.codecogs.com/gif.latex?x" title="x" /> the one-hot encoded input, <img src="https://latex.codecogs.com/gif.latex?h" title="h" /> the hidden embedded representation and <img src="https://latex.codecogs.com/gif.latex?W_1,&space;\;&space;W_2" title="W_1, \; W_2" /> respectively the encoding and decoding matrices.

### Optimising the model
As asked, the model can be optimised, particularly for big datasets.

Thus, to cope with the memory issues we had to:
- change the matrices from Numpy arrays to Scipy sparse matrices.
- change the data types of the matrices (defaults are float64 or int64) so we set to the minimum (float16 and int8).

To speed up the computation, negative sampling is a good solution as it approximates the Softmax operation to avoid having to sum over the whole vocabulary which is very computationally expensive. For each example, a different set of negative words is sampled, even in a same batch: hence we get more diversity especially since, in our model, a center word is present in different lines of the training matrix (it's not regrouped), thus for a unique center word, different context words are used. It helped getting more speed and being able to train on bigger dataset. To sample negative data, more frequent words should be more selected but not using simple word frequencies, instead we implemented the normalizing techniques of Word2Vec because it seems to work better; thus the probability of sampling a word becomes:

<img src="https://latex.codecogs.com/gif.latex?P(w_i)&space;=&space;\frac{f(w_i)^{3/4}]}{\sum_{j=0}^{n}f(w_j)^{3/4}}" title="P(w_i) = \frac{f(w_i)^{3/4}]}{\sum_{j=0}^{n}f(w_j)^{3/4}}" />

### Difficulties
A big issue we faced and that we spent a very long time on was the exploding gradient problem which made the exponential computation in the softmax overflow, even with negative sampling. Thus, managed to solve this problem by putting a saturation before the exponential (values are in [-10, 10]) and we performed gradient clipping to prevent it from exploding.

Besides, the model has to be trained on an enormous amount of data to display good results, thus it must undergo a very long phase of training (more than a day) which is very hard to perform analysis, optimizing parameters or just tests.

## Sources
- [1](http://cs231n.github.io/neural-networks-3/): Stanford 231n course additional material: used for good techniques to verify a neural network from scratch is well implemented (Ex: compare the analytical and numerical gradients)
- [2](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) Blog post explaining Word2Vec and the skip-gram model as well as what is negative sampling.
- [3](https://github.com/chrisjmccormick/word2vec_commented) Original Word2Vec C code commented to get ideas of how it was implemented.
- [4](http://building-babylon.net/2015/07/13/word2vec-weight-initialisation/) Initialisation used in Word2Vec for the layers
- [5](https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/), [6](https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72), [7](https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281) Blogs about implementing Word2Vec from scratch in Python: we took some inspirations to build our model (especially the computation of the gradients).
- [8](https://www.cs.bgu.ac.il/~yoavg/publications/negative-sampling.pdf) Paper from the given ressources explaining in depth what is negative sampling.
- [9](https://arxiv.org/abs/1301.3781) Word2Vec original paper
