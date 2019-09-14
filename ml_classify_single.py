import sys
import numpy as np
from utils_preprocess_title import preprocess_title
from keras.preprocessing.sequence import pad_sequences
import keras
import os
DICTIONARY_OUTPUT_DIR = './output/dictionary.npy'
MODEL = '0.8699.hdf5'

dictionary = np.load(DICTIONARY_OUTPUT_DIR, allow_pickle=True)
categories = np.load('./output/categories.npy', allow_pickle=True)

vocabulary_size = dictionary.shape[0] + 1
dic_list = list(dictionary)
max_sequence = 24

def normalize(title, language):
  print("Processing: " + title)
  tokens = preprocess_title(title, language)
  tokens = [(dic_list.index(token) + 1) if token in dic_list else (len(dic_list) + 2) for token in tokens]
  tokens = pad_sequences([tokens], maxlen=max_sequence, padding='post')[0]
  return tokens

model = keras.models.load_model(os.path.join('./models/', MODEL))

text = normalize(sys.argv[1], "portuguese")
pred = model.predict(np.array([text]))
print(categories[np.argmax(pred[0])])

