import numpy as np
from multiprocessing import Pool
from hash import hash_tokens
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
tokens = np.load('./titles.npy', allow_pickle=True)
print(tokens.shape)
dictionary = np.load('./dic.npy', allow_pickle=True)
print("Max word sequence: ")
max_sequence = max(len(l) for l in tokens)
max_sequence_plus = max_sequence + 1
print(max_sequence)

vocabulary_size = dictionary.shape[0]
chunks = np.array_split(tokens, 8)

def _hash(arr):
  hashed_list = []
  for item in tqdm(arr):
    hashed_list.append(hash_tokens(item, vocabulary_size))
  return pad_sequences(hashed_list, maxlen=max_sequence_plus, padding='post')

with Pool(10) as p:
  data = p.map(_hash, chunks)
  hashed_list = np.concatenate([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]])
  #for i in range(8):
  #hashed_list = hashed_list + data[i]
  np.save('./hashed.npy', hashed_list)
  print(hashed_list.shape)
