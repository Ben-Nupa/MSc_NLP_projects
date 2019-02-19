# NLP Homework

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