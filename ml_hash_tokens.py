import numpy as np
from multiprocessing import Pool
from utils_hash import hash_tokens
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

tokens = np.load('./titles_normal.npy', allow_pickle=True)
#tokens = tokens[0:30]
dictionary = np.load('./dic.npy', allow_pickle=True)

print("Max word sequence: ")
max_sequence = max(len(l) for l in tokens)
print(max_sequence)
max_sequence = max_sequence * 2
vocabulary_size = dictionary.shape[0] + 1 # to add undefinedword
chunks = np.array_split(tokens, 8)
dic_list = list(dictionary)

def _hash(arr):
  hashed_list = []
  for item in tqdm(arr):
    hashed_list.append(hash_tokens(item, vocabulary_size))
  #return pad_sequences(hashed_list, maxlen=max_sequence, padding='post')
  return hashed_list
with Pool(10) as p:
  data = p.map(_hash, chunks)
  hashed_list = np.concatenate([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]])
  np.save('./hashed.npy', hashed_list)
  print(hashed_list.shape)
  print(hashed_list)
