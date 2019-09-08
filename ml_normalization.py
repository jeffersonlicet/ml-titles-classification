import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from ml_preprocess_title import preprocess_title
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, shm_size_mb=int(32e3))

TRAIN_CSV_DIR = './train.csv'
CATEGORIES_OUTPUT_DIR = './categories.npy'
TITLES_OUTPUT_DIR = './titles.npy'
DICTIONARY_OUTPUT_DIR = './dictionary.npy'
LABELS_OUTPUT_DIR = './labels.npy'

data = pd.read_csv(TRAIN_CSV_DIR)
#data = data.sample(5)
print("There are: " + str(data.shape[0]) + " entries in the csv file")

unreliable = data[data['label_quality'] == 'unreliable']
reliable = data[data['label_quality'] == 'reliable']

print("There are: " + str(data.shape[0]-(unreliable.shape[0] + reliable.shape[0])) + " invalid entries")

merged = pd.concat([
  unreliable,
  reliable,
])

merged = merged.drop('label_quality', axis=1)
#print(merged.head())
categories = merged.category.unique()
categories = np.sort(categories)

print("Total categories: " + str(categories.shape[0]))
np.save(CATEGORIES_OUTPUT_DIR, categories)

categories_list = list(categories)
dictionary = Counter()

tokens_list = []
categories = []
def normalize(row):
  tokens = preprocess_title(row.title, row.language)
  category = categories_list.index(row.category)
  categories.append(category)
  dictionary.update(tokens)
  row.title = tokens
  row.category = category
  
  return row

tokens = merged.parallel_apply(normalize, axis=1)
tokens_list = np.squeeze(tokens.iloc[:,0:1].values)
categories = np.squeeze(tokens.iloc[:,2:3].values)
print("Mean by word")
mean = np.mean([c for k,c in dictionary.items()])
print(mean)

dictionary_list = [k for k,c in dictionary.items()]

print("Total words in dictionary")
print(len(dictionary_list))

np.save(TITLES_OUTPUT_DIR, tokens_list)
print(tokens_list)
print("The titles are now normalized")
np.save(DICTIONARY_OUTPUT_DIR, dictionary_list)
print("The Dictionary has been generated")
np.save(LABELS_OUTPUT_DIR, categories)
print("The labels are now separated")
