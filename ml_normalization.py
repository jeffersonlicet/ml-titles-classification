import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils_preprocess_title import preprocess_title
from multiprocessing import  Pool
import multiprocessing

TRAIN_CSV_DIR = './data/train.csv'
CATEGORIES_OUTPUT_DIR = './output/categories.npy'
TITLES_OUTPUT_DIR = './output/titles_normal.npy'
LABELS_OUTPUT_DIR = './output/labels.npy'

data = pd.read_csv(TRAIN_CSV_DIR)

print("There are: " + str(data.shape[0]) + " entries in the csv file")

unreliable = data[data['label_quality'] == 'unreliable']
reliable = data[data['label_quality'] == 'reliable']

print("There are: " + str(data.shape[0]-(unreliable.shape[0] + reliable.shape[0])) + " invalid entries")

merged = pd.concat([
  unreliable,
  reliable,
])

merged = merged.drop('label_quality', axis=1)
categories = merged.category.unique()
categories = np.sort(categories)

print("Total categories: " + str(categories.shape[0]))
np.save(CATEGORIES_OUTPUT_DIR, categories)

categories_list = list(categories)

tokens_list_stemmed = []
tokens_list = []
categories = []

def normalize(row):
  tokens = preprocess_title(row.title, row.language)
  category = categories_list.index(row.category)
  row.title = tokens
  row.category = category
  return row

def normalize_chunk(data):
  return data.apply(normalize, axis=1)

def norm(dataframe):
  WORKERS = multiprocessing.cpu_count()
  with Pool(WORKERS) as p:
    df_split = np.array_split(dataframe, WORKERS)
    df = pd.concat(p.map(normalize_chunk, df_split))
    return df

tokens = filtered = norm(merged)
tokens = tokens[tokens.title.str.len() != 0]

tokens_list = np.squeeze(tokens.iloc[:,0:1].values)
categories = np.squeeze(tokens.iloc[:,2:3].values)

np.save(TITLES_OUTPUT_DIR, tokens_list)
print("The titles are now normalized")

np.save(LABELS_OUTPUT_DIR, categories)
print("The labels are now separated")

print(tokens_list[0:10])
