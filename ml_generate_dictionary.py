import numpy as np
from tqdm import tqdm
from collections import Counter

titles = np.load('./output/titles_normal.npy', allow_pickle=True)
dictionary = Counter()

for tokens in tqdm(titles):
  dictionary.update(tokens)

print("Mean by word")
mean = np.mean([c for k,c in dictionary.items()])
print(mean)

print("Total words found:")
print(len(dictionary.items()))

dictionary_list = [k for k,c in dictionary.items() if c >= 2]

print("Total words in dictionary")
print(len(dictionary_list))

np.save('./output/dictionary.npy', dictionary_list)
print("The Dictionary has been generated")
