import numpy as np
from collections import Counter
from tqdm import tqdm
titles = np.load('./output_beta/titles_normal.npy', allow_pickle=True)

print("Titles")
print(titles.shape)

dictionary = Counter()

for tokens in tqdm(titles):
  dictionary.update(tokens)
  #print(tokens)
  #break
print("Mean by word")
mean = np.mean([c for k,c in dictionary.items()])
print(mean)
print("Total words found:")
print(len(dictionary.items()))
dictionary_list = [k for k,c in dictionary.items() if c >= 5]

print("Total words in dictionary")
print(len(dictionary_list))

np.save('./output_beta/dictionary.npy', dictionary_list)
#print(dictionary_list)
print("The Dictionary has been generated")
