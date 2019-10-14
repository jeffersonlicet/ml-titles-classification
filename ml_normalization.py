import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import  Pool
from utils_preprocess_title import preprocess_title

TRAIN_CSV_DIR = './data/train.csv'
CATEGORIES_OUTPUT_DIR = './output/categories.npy'
TITLES_OUTPUT_DIR = './output/titles_normal.npy'
LABELS_OUTPUT_DIR = './output/labels.npy'

data = pd.read_csv(TRAIN_CSV_DIR)

unreliable = data[data['label_quality'] == 'unreliable']
reliable = data[data['label_quality'] == 'reliable']

merged = pd.concat([
  unreliable,
  reliable,
])

merged = merged.drop('label_quality', axis=1)
categories = merged.category.unique()
categories = np.sort(categories)

np.save(CATEGORIES_OUTPUT_DIR, categories)

categories_list = list(categories)
dictOfCategories = { word : i for i, word in enumerate(categories_list) }

tokens_list_stemmed = []
tokens_list = []
categories = []

def normalize(row):
  row.title = preprocess_title(row.title, row.language)
  row.category = dictOfCategories.get(row.category)
  return row

def normalize_chunk(data):
  tqdm.pandas()
  return data.progress_apply(normalize, axis=1)

def norm(dataframe):
  WORKERS = multiprocessing.cpu_count()
  with Pool(WORKERS) as p:
    df_split = np.array_split(dataframe, WORKERS)
    df = pd.concat(p.map(normalize_chunk, df_split))
    return df

tokens = filtered = norm(merged)
tokens = tokens[tokens.title.str.len() != 0]

tokens_list = np.squeeze(tokens.iloc[:, 0:1].values)
categories = np.squeeze(tokens.iloc[:, 2:3].values)

np.save(TITLES_OUTPUT_DIR, tokens_list)
np.save(LABELS_OUTPUT_DIR, categories)
