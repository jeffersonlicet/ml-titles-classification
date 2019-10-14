import os
import csv
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from multiprocessing import Pool

MODEL = sys.argv[1]

test = np.load('./output/test_titles.npy', allow_pickle=True)
categories = np.load('./output/categories.npy', allow_pickle=True)

print("Total test items")
print(test.shape)

chunks = np.array_split(test, 8)

def predict(elements):
  model = tf.keras.models.load_model(os.path.join('./models/', MODEL))
  predictions = []
  pred = model.predict(elements)
  for p in pred:
    predictions.append(categories[np.argmax(p)])
  return predictions

with Pool(1) as p:
  result = np.concatenate(p.map(predict, chunks))
  print("Total items predicted: ")
  print(result.shape)
  with open('./submission_'+MODEL+'.csv', mode='w') as base_file:
    csv_iterator = csv.writer(base_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_iterator.writerow(['id', 'category'])
    for i, value in enumerate(result):
      csv_iterator.writerow([i, value])
