import numpy as np
import keras
import os
import csv
from multiprocessing import Pool
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
MODEL = '8-weights-improvement.hdf5'

test = np.load('./test_titles.npy', allow_pickle=True)
categories = np.load('./categories.npy', allow_pickle=True)

print("Total test items")
print(test.shape)
print(test[0])
print(test[1])
print(test[9:15])
exit()
chunks = np.array_split(test, 8)


def predict(elements):
  model = keras.models.load_model(os.path.join('./models/', MODEL))
  predictions = []
  for element in tqdm(elements):
    pred = model.predict(np.array([element]))
    predictions.append(categories[np.argmax(pred[0])])
  return predictions

with Pool(1) as p:
  data = p.map(predict, chunks)
  result = np.concatenate([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]])
  print("Total items predicted: ")
  print(result.shape)
  with open('./submission2.csv', mode='w') as base_file:
    csv_iterator = csv.writer(base_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_iterator.writerow(['id', 'category'])
    for i, value in enumerate(result):
      csv_iterator.writerow([i, value])
