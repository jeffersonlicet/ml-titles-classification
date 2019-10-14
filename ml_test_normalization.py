import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from pandarallel import pandarallel
from utils_preprocess_title import preprocess_title
from keras.preprocessing.sequence import pad_sequences

pandarallel.initialize(progress_bar=True)

TEST_CSV_DIR = './data/test.csv'
DICTIONARY_OUTPUT_DIR = './output/dictionary.npy'

data = pd.read_csv(TEST_CSV_DIR)

max_sequence = 27

dictionary = np.load(DICTIONARY_OUTPUT_DIR, allow_pickle=True)
vocabulary_size = dictionary.shape[0] + 1
dic_list = list(dictionary)

print("There are: " + str(data.shape[0]) + " entries in the csv file")

def normalize(row):
  tokens = preprocess_title(row.title, row.language)
  row.title = [(dic_list.index(token) + 1) if token in dic_list else 0 for token in tokens]
  return row

tokens = data.parallel_apply(normalize, axis=1)
tokens_list = np.squeeze(tokens.iloc[:,1:2].values)
tokens_list = pad_sequences(tokens_list, maxlen=max_sequence, padding='post')

np.save('./output/test_titles.npy', tokens_list)

print(tokens_list.shape)
print(tokens_list)

