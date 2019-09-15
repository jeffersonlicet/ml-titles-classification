import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from utils_preprocess_title import preprocess_title
from pandarallel import pandarallel
from utils_hash import hash_tokens
from keras.preprocessing.sequence import pad_sequences

pandarallel.initialize(progress_bar=True, shm_size_mb=int(2e3))

UNDEFINED_WORD = 'undefinedword'
TEST_CSV_DIR = './data/test.csv'
DICTIONARY_OUTPUT_DIR = './output/dictionary.npy'

data = pd.read_csv(TEST_CSV_DIR)
#data = data.sample(10)
must_sum = (data.shape[0]-1) * (data.shape[0]) / 2
total_sum = data.id.sum()

print("Must: " + str(must_sum))
print("Items id sums: " + str(total_sum))
print("Items are sorted: " + str(must_sum == total_sum))

#hashed = np.load('./hashed.npy', allow_pickle=True)

max_sequence = 24
print("Max sequence")
print(max_sequence)

dictionary = np.load(DICTIONARY_OUTPUT_DIR, allow_pickle=True)
vocabulary_size = dictionary.shape[0] + 1 # to add undefinedword
dic_list = list(dictionary)

print("There are: " + str(data.shape[0]) + " entries in the csv file")
tokens_list = []

def normalize(row):
  tokens = preprocess_title(row.title, row.language)
  #filtered = hash_tokens(tokens, vocabulary_size)
  filtered = [(dic_list.index(token) + 1) if token in dic_list else (len(dic_list) + 2) for token in tokens]
  row.title = filtered
  return row

print("Starting")
tokens = data.parallel_apply(normalize, axis=1)

tokens_list = np.squeeze(tokens.iloc[:,1:2].values)
tokens_list = pad_sequences(tokens_list, maxlen=max_sequence, padding='post')

np.save('./output/test_titles.npy', tokens_list)

print(tokens_list.shape)
print(tokens_list)

