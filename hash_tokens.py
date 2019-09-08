import numpy as np
from multiprocessing import Pool
from hash import hash_tokens
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
tokens = np.load('./titles.npy', allow_pickle=True)
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
  hashed_list =np.squeeze(np.array(data))
  #for i in range(8):
  #hashed_list = hashed_list + data[i]
  np.save('./hashed.npy', hashed_list)

