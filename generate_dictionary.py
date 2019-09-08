import numpy as np
from collections import Counter
from tqdm import tqdm
titles = np.load('./titles.npy', allow_pickle=True)

print("Titles")
print(titles.shape)

dictionary = Counter()

for tokens in tqdm(titles):
  dictionary.update(tokens)

print("Mean by word")
mean = np.mean([c for k,c in dictionary.items()])
print(mean)

dictionary_list = [k for k,c in dictionary.items()]

print("Total words in dictionary")
print(len(dictionary_list))

np.save('./dic.npy', dictionary_list)
print("The Dictionary has been generated")
