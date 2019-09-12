import numpy as np

labels = np.load('./output/labels.npy', allow_pickle=True)
print("Labels")
print(labels.shape)
#print(labels[0:5])
categories = np.load('./output/categories.npy', allow_pickle=True)
print("Categories")
print(categories.shape)
#for label in labels[0:5]:
#  print(categories[label])
titles =  np.load('./output/titles_normal.npy', allow_pickle=True)
#print(titles[0:5])
print("Titles")
print(titles.shape)
hashed =  np.load('./output/hashed.npy', allow_pickle=True)
dictionary = np.load('./output/dictionary.npy', allow_pickle=True)
#categories = list(categories)
choice = np.random.choice(labels, size=100, replace=False)
for i in choice:
  label = labels[i]
  title = titles[i]
  print(categories[label])
  print(title)
  print(hashed[i])
  print([dictionary[k-1] if k <= len(dictionary) else 'undefined' for k in hashed[i]])
  print('\n')
