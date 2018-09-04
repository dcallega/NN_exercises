# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

print(tf.__version__)
checkpoint_dir = "training_1/"
checkpoint_path = checkpoint_dir + "cp.ckpt"
print("Saving last checkpoint in: {}".format(checkpoint_path))

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(9,)),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])
  
  return model

# If no function is passed, makes a fake dataset with 9 features in [0,1] and objective function
#   SUM(feats) > K.
# Note: each row is one example!
# Note2: remember to NORMALIZE each feature. Each feature should more or less have
#   the same interval (suggested [0,1])

def load_dataset(file=None):
  if file is None:
    X = np.random.rand(10000, 9)
    y = np.sum(X, axis=1)
    for i in range(len(y)):
      if y[i] > 3:
        y[i] = 1
      else:
        y[i] = 0
  else:
    raise NotImplementedError
  return X, y

def custom_fit(model, X_tr, y_tr, X_val, y_val):
  logs_dir = os.path.join(checkpoint_dir, 'board_logs')

  # Create checkpoint callback
  cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                   save_weights_only=True,
                                                   verbose=1)
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, 
                                               histogram_freq=0,  
                                               write_graph=True)

  model.fit(X_train, y_train, epochs = 25, 
            validation_data = (X_val,y_val),
            callbacks = [cp_callback, tb_callback])  # pass callback to training

  return model


TRAIN = True
if __name__=="__main__":
  print("BEFORE RUNNING script, run command:\n mkdir training_1 && mkdir training_1/board_logs")
  X, y = load_dataset()

  # Split dataset in train, validation and test sets
  X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.33, random_state=42)

  print("Training set dims:", X_train.shape, y_train.shape)

  if TRAIN:
    m = create_model()
    m_fit = custom_fit(m, X_train, y_train, X_val, y_val)

    print(m_fit.summary())
    m_fit.save('training_1/my_model.h5')

    del m, m_fit

  m = create_model()
  m_fit = keras.models.load_model('training_1/my_model.h5')

  loss, acc = m.evaluate(X_test, y_test)
  print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

  loss, acc = m_fit.evaluate(X_test, y_test)
  print("Restored model, accuracy: {:5.2f}%".format(100*acc))

  m.load_weights(checkpoint_path)
  loss, acc = m.evaluate(X_test, y_test)
  print("Restored model, accuracy: {:5.2f}%".format(100*acc))


  sum5 = np.array([[0.5, 0.5, 0.8, 0.4, 0.2, 0.9, 0.5, 0.7, 0.5], ])#.transpose()
  print(sum5.shape)

  print(m.predict(sum5))
  print("Check the training running from the current directory: ")
  print("tensorboard --logdir training_1/board_logs")


