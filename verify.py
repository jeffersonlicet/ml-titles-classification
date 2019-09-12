import numpy as np

labels = np.load('./labels.npy', allow_pickle=True)
print("Labels")
print(labels.shape)
#print(labels[0:5])
categories = np.load('./categories.npy', allow_pickle=True)
print("Categories")
print(categories.shape)
#for label in labels[0:5]:
#  print(categories[label])
titles =  np.load('./titles_normal.npy', allow_pickle=True)
#print(titles[0:5])
print("Titles")
print(titles.shape)
#categories = list(categories)
choice = np.random.choice(labels, size=100, replace=False)
for i in choice:
  label = labels[i]
  title = titles[i]
  print(categories[label])
  print(title)
