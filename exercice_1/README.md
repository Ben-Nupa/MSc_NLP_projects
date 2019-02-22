# Exercise 1: Skip-gram with negative-sampling from scratch

## Run

To train, type:
```bash
python3 skipGram.py --text data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled --model model.pkl
```
To test, type:
```bash
python3 skipGram.py --text data/EN-SIMLEX-999.txt --model model.pkl --test
```
If the similarity score given is `-1`, this means we didn't had this word in our vocabulary.

## Pipeline
One-hot encoding of data:
- Read file line by line until the desired number of lines is achieved
- Data filtering : set to lowercase, remove all non alphanumerical characters except spaces and ', transform ' to spaces
- Map each word to an ID and vice versa in a dictionnary
- Generate the datasets in 2 different forms: list of indices of words and one-hot sparse encoded matrices. The i-th element of the X list corresponds to a center word and the i-th element of the Y list corresponds to a context word of this center word. For instance if the center word '20' has the contex words (let's say window size of 3) '52', '38', '11', '0', '124', '89'; we'll have `X = [..., 20, 20, 20, 20, 20, 20, ...]` and `Y = [..., 52, 38, 11, 0, 124, 89, ...]`.

For the training:
- Weights initialization: Uniform law over <a href="https://www.codecogs.com/eqnedit.php?latex=[\frac{-1}{2n_{embed}},&space;\frac{1}{2n_{embed}}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[\frac{-1}{2n_{embed}},&space;\frac{1}{2n_{embed}}]" title="[\frac{-1}{2n_{embed}}, \frac{1}{2n_{embed}}]" /></a> for the encoding W1 matrix and 0-matrix for the decoding W2 matrix.
- Training with batch (a center word can be in several batches) with negative sampling for the softmax approximation, saturation on exponential to avoid overflow, gradient clipping to prevent it from exploding and SGD algorithm for the update.
- By default, the learning rate is decayed (default factor is 0.99) after each epoch and the total loss is computed: the learning curve is plot at the end of the training.

Similarity

## Thought process
The skip-gram model is simply a fully-connected neural network with one hidden layer without activation function, thus we built a class for this general kind of networks. A focus was made to compute an analytical gradient so as to make computation as effective as possible (numpy matrices-based operations and no 'for loop'): to check that the gradient was correctly computed, we compared it to the numerical gradient computed with the central differencing scheme. The following operations are applied:
OPERATIONS TO WRITE

## Sources

## TODO
- Function to read all the files, one by one.  
- Then, we need to finish the negative sampling part (already begun by Benoit)  
- Training the network : try different architecture (linear model, no activation function for the hidden layer)  
- Function to store the model : Paul  
- Ways to increase the training speed  

## Matrix creation
x & y :  
x = [id_center_word_0, id_center_word_1, ...]  
y = [id_context_word_0, id_context_word_1, ...]  

Then we create a one-hot matrix X, of size vocab_size * dataset_size (really sparse matrix)  
For each center-word, this matrix gives us the context  

## Negative sampling

Negative sampling : when we compute softmax (exp(v)/sum(exp(v))) : computing it is expensive because of huge vocabulary size.  
Thus, we compute this only on the word over "negative word with samples".
We don't want to go on each word of our 1M vector, we sample randomly words, we test if they are not in the the batch.  
Y_bath_indices UNION Negative_sampling_indices : check if intersection is empty. If yes, continue with back propagation.  
Batch size ~ 256 max, this way we should have a empty intersection almost each time. If not empty, we use a random new samples in the paper the distribution.  

Drawbacks : INPUT/OUTPUT is center-word -> context words. But on next bath, we can have other context words (for the same word ?). We try not to pick the same words from the context word selection.  