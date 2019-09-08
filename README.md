Run ml_normalization.py to generate
  categories.npy
  titles.npy
  labels.npy

Run ml_generate_dictionary.py to generate
  dic.npy

Run ml_hash_tokens to generate
  hashed.npy

Run ml_train.py to train the model using
  hashed.npy titles as x data
  labels.npy as y data
  dict.npy to get vocabulary

Run ml_test_normalization to normalize test data and generate
  test_titles.npy