import numpy as np
from utils_hash import hash_tokens
from tqdm import tqdm

embeddings = np.load('./embeddings/on-dic-fast.pkl', allow_pickle=True)

dictionary = np.load('./dic.npy', allow_pickle=True)
vocabulary_size = dictionary.shape[0] + 1

embedding_matrix = np.zeros((vocabulary_size, 300))
print("Processings vocab: " + str(len(embeddings.items())))

words = list(embeddings.keys())
print("Total word on embeddings: ")
print(len(words))

hashes = hash_tokens(words, vocabulary_size)
print("Hashes len")
print(len(hashes))

i = 0
for word, vector in tqdm(embeddings.items()):
  embedding_matrix[hashes[i]] = vector
  i = i + 1

np.save('./embeddings/matrix_fast.npy', embedding_matrix)
print("Embeddings matrix saved")
print(embedding_matrix)
