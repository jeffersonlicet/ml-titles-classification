import os
import numpy as np
import random as rn
from pathlib import Path
import tensorflow.keras as keras
from tensorflow import set_random_seed
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

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

x, x_test, y, y_test = train_test_split(x, y, test_size=0.001, random_state=10, stratify=y)
print('unique test')
print(len(np.unique(y_test)))
class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
print('Class weights')
print(class_weights)

EMBEDDINGS_DIMENSION = 256

model = keras.models.Sequential()
model.add(keras.layers.Embedding(
  trainable=True,
  input_length=x.shape[1],
  output_dim=EMBEDDINGS_DIMENSION,
  input_dim=(dictionary.shape[0]+2)
))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(categories_list), activation='softmax'))

model.compile(
  loss=keras.losses.sparse_categorical_crossentropy,
  optimizer=keras.optimizers.Adam(),
  metrics=['accuracy']
)

print(model.summary())
BATCH_SIZE = 2**15

callbacks_list = []

history = model.fit(x, y, epochs=18, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), class_weight=dict(enumerate(class_weights)), callbacks=callbacks_list, verbose=2)
model.save('model.h5')