import numpy as np
from multiprocessing import Pool
from utils_hash import hash_tokens
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

tokens = np.load('./output/titles_normal.npy', allow_pickle=True)
dictionary = np.load('./output/dictionary.npy', allow_pickle=True)

WORKERS = 16

print("Max word sequence: ")
max_sequence = max(len(l) for l in tokens)
print(max_sequence)

vocabulary_size = dictionary.shape[0] + 1
chunks = np.array_split(tokens, WORKERS)
dic_list = list(dictionary)
dictOfWords = { word : i for i, word in enumerate(dic_list) }

def _hash(arr):
  hashed_list = []
  for item in tqdm(arr):
    _list = []
    for token in item:
      try:
        _list.append((dictOfWords.get(token, -1) + 1))
      except:
        _list.append(0)
    hashed_list.append(_list)
  return hashed_list
with Pool(WORKERS) as p:
  data = p.map(_hash, chunks)
  hashed_list = data[0]
  for i in range(1, WORKERS):
    hashed_list = np.concatenate([hashed_list, data[i]])
  np.save('./output/hashed.npy', hashed_list)
  print(hashed_list.shape)
  print(hashed_list[0])


