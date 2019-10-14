
### Public Leaderboard Top 20 - 0.89546

### The challenge

The challenge was very interesting, classify articles using only its titles.

### The prize

First and second place will receive tickets to [KHIPU](https://khipu.ai/). From 3 to 5 place an Intel Movidius.

### The result

I managed to be in the top 20 with a score of 0.8954, there were more of 150 participants, the competition was hard and exciting. Of course, i learned a lot of things.



## Preprocessing

Run ml_normalization.py to generate
 - categories.npy
 - titles.npy
 - labels_normal.npy

  
## Generate Dictionary
Run ml_generate_dictionary.py to generate
 - dictionary.npy

  
## Transform tokens
Run ml_index_tokens to generate
 - hashed.npy

## Training the model
Run ml_train.py to train the model using
 - hashed.npy titles as x data
 - labels.npy as y data
 - dictionary.npy to get vocabulary

 
## Preprocess test data
Run ml_test_normalization.py to normalize test data and generate
 - test_titles.npy

 
## Generate submission file
Run ml_classify.py to generate the submission file