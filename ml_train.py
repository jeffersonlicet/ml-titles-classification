import numpy as np
import keras

from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
import random as rn
import os
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss
from sklearn.utils import class_weight

# Seed to get reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
set_random_seed(2)

categories = 1588

hash_list = np.load('./output/hashed.npy', allow_pickle=True)
labels = np.load('./output/labels.npy', allow_pickle=True)
dictionary = np.load('./output/dictionary.npy', allow_pickle=True)

print(hash_list.shape)
#exit()
#embedding_matrix = np.load('./embeddings/matrix_fast.npy', allow_pickle=True)
#embeddings = np.load('./embeddings/on-dic-fast.pkl', allow_pickle=True)
print("Vocabulary size: ")
print(len(dictionary))
#words = list(embeddings.keys())
#print("Total word with embeddings: ")
#print(len(words))
#hash_list = hash_list[0:100]
#labels = labels[0:100]

filepath = "./models/8-weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

output_dim = 4

print("DATA")
print(hash_list.shape)
print(hash_list)
print(labels)
print(labels.shape)

def generator(X_data, y_data, batch_size):
  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter=0

  while 1:
    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch, y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0
maxlen = 24
#x, x_test, y, y_test = train_test_split(hash_list, labels, test_size=0.2, random_state=4)
hash_list = pad_sequences(hash_list, maxlen=maxlen, padding='post')
#x_test = pad_sequences(x_test, maxlen=26, padding='post')
x, x_test, y, y_test = train_test_split(hash_list, labels, train_size=0.1, test_size=0.01, random_state=4, stratify=labels)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y)
print("Class weights: ")
print(class_weights)
print(x.shape)
print(y.shape)
print(x[1])
print(y[1])
model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=(dictionary.shape[0]+1), output_dim=output_dim, input_length=x.shape[1], trainable=True))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(2048, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1588, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print(model.summary())
BATCH_SIZE = 4000
#training_generator = BalancedBatchGenerator(hash_list, labels, sampler=NearMiss(), batch_size=BATCH_SIZE, random_state=42)
#validation_generator = BalancedBatchGenerator(x_test, y_test, sampler=NearMiss(), batch_size=BATCH_SIZE, random_state=42)

#training_generator = generator(x, y, BATCH_SIZE)
#validation_generator = generator(x_test, y_test, BATCH_SIZE)

#print("Steps per epoch: ")
#steps_per_epoch = x.shape[0]/BATCH_SIZE
#print(steps_per_epoch)

#print("Validation steps: ")
#validation_steps = x_test.shape[0]/BATCH_SIZE
#print(validation_steps)
"""
history = model.fit_generator(
  training_generator,
  epochs=10,
  #validation_data=validation_generator,
  callbacks=callbacks_list,
  #steps_per_epoch=steps_per_epoch,
  #validation_steps=validation_steps,
  use_multiprocessing=True,
  workers=4,
)
"""
#history = model.fit(x, y, epochs=100, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=callbacks_list)
# evaluate
#loss, acc = model.evaluate(x, y)
##print('Train Accuracy: %f' % (acc*100))
history = model.fit(x, y, epochs=100, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=callbacks_list, class_weight=class_weights)
oss, acc = model.evaluate(x, y)
print('Train Accuracy: %f' % (acc*100))

loss, acc = model.evaluate(x_test, y_test)
print('Test Accuracy: %f' % (acc*100))

model.save('./models/model_'+str(acc*100)+'_.h5')
