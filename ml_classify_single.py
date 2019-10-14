import sys
import numpy as np
from utils_preprocess_title import preprocess_title
from keras.preprocessing.sequence import pad_sequences
import keras
import os

DICTIONARY_OUTPUT_DIR = './output/dictionary.npy'
MODEL = sys.argv[1]

dictionary = np.load(DICTIONARY_OUTPUT_DIR, allow_pickle=True)
categories = np.load('./output/categories.npy', allow_pickle=True)

vocabulary_size = dictionary.shape[0] + 1
dic_list = list(dictionary)
max_sequence = 27

def normalize(title, language):
  print("Processing: " + title)
  tokens = preprocess_title(title, language)
  tokens = [(dic_list.index(token) + 1) if token in dic_list else 0 for token in tokens]
  tokens = pad_sequences([tokens], maxlen=max_sequence, padding='post')[0]
  return tokens

model = keras.models.load_model(os.path.join('./models/', MODEL))

text = normalize(sys.argv[2], sys.argv[3])
pred = model.predict(np.array([text]))

print(categories[np.argmax(pred[0])])
