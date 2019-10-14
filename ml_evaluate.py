import os
import keras
import numpy as np
import random as rn
from pathlib import Path
from sklearn.utils import class_weight
from tensorflow import set_random_seed
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import sys

MODEL = sys.argv[1]

# Seed to get reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
set_random_seed(2)

BASE_PATH = Path('./')
ASSEMBLED_DIR = BASE_PATH / 'output'

x = np.load(ASSEMBLED_DIR / 'hashed.npy', allow_pickle=True)

y = np.load(ASSEMBLED_DIR / 'labels.npy', allow_pickle=True)

print('Total categories on y: ')
print(len(np.unique(y)))

dictionary = np.load(ASSEMBLED_DIR / 'dictionary.npy', allow_pickle=True)

maxlen = max(len(l) for l in x)
print('Max sequence length: ' + str(maxlen))

categories = np.load(ASSEMBLED_DIR / 'categories.npy', allow_pickle=True)
categories_list = list(categories)

x = pad_sequences(x, maxlen=maxlen)

x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=4, stratify=y)

model = keras.models.load_model('./models/'+MODEL)

y_pred = model.predict(x_test, verbose=0)
y_pred_max = np.argmax(y_pred, axis=1).tolist()

print("Balanced Acc for: ")
bacc = balanced_accuracy_score(y_test, y_pred_max)
print ("########## Balanced Acc: %0.8f ##########" % bacc )
