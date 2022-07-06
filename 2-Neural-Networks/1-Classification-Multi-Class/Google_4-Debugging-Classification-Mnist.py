from os.path import join # for joining file pathnames
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import unittest
import sys

# Set Pandas display options
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load data
mnistDf_backup = pd.read_csv(
  "https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv",
  sep=",",
  header=None)
# Shuffle data
mnistDf_backup.sample(frac=1).reset_index(drop=True)
# Use the first 5000 examples for faster prototyping
mnistDf = mnistDf_backup[0:5000]

mnistDf.head()

# Calculate the number of classes
numClasses = mnistDf.iloc[:,0].unique().shape[0]
# Plot histogram of class distribution to see if we have an imbalanced data
plt.hist(mnistDf.iloc[:,0], bins=range(numClasses+1))
plt.xticks(range(numClasses+1))

# Shuffle data
mnistDf = mnistDf.sample(frac=1).reset_index(drop=True)

# Split dataset into data and labels
mnistData = mnistDf.iloc[:,1:-1].copy(deep=True)
mnistLabels = mnistDf.iloc[:,0].copy(deep=True)

# Process data
# Define the scaling function to check for zeros
def minMaxScaler(arr):
  max = np.max(arr)
  if(max!=0):  # avoid /0
    min = np.min(arr)
    arr = (arr-min)/max
  return arr

# Scale data
for featureIdx in range(mnistData.shape[1]):
  mnistData.iloc[:,featureIdx] = minMaxScaler(mnistData.iloc[:,featureIdx])

mnistData.describe()

# Establish baseline
np.sum(mnistLabels==1)*1.0/mnistLabels.shape[0]*100

# Define a plot function
def showClassificationResults(trainHistory):
  """Function to:
   * Print final loss & accuracy.
   * Plot loss & accuracy curves.

  Args:
    trainHistory: object returned by model.fit
  """

  # Print final loss and accuracy
  print("Final training loss: " + str(trainHistory.history['loss'][-1]))
  print("Final validation loss: " + str(trainHistory.history['val_loss'][-1]))
  print("Final training accuracy: " + str(trainHistory.history['accuracy'][-1]))
  print("Final validation accuracy: " + str(trainHistory.history['val_accuracy'][-1]))

  # Plot loss and accuracy curves
  f = plt.figure(figsize=(10,4))
  axLoss = f.add_subplot(121)
  axAcc = f.add_subplot(122)
  axLoss.plot(trainHistory.history['loss'])
  axLoss.plot(trainHistory.history['val_loss'])
  axLoss.legend(['Training loss', 'Validation loss'], loc='best')
  axLoss.set_xlabel('Training epochs')
  axLoss.set_ylabel('Loss')
  axAcc.plot(trainHistory.history['accuracy'])
  axAcc.plot(trainHistory.history['val_accuracy'])
  axAcc.legend(['Training accuracy', 'Validation accuracy'], loc='best')
  axAcc.set_xlabel('Training epochs')
  axAcc.set_ylabel('Accuracy')

###############################################

# Train a linear model
model = None
# Define
model = keras.Sequential()
model.add(keras.layers.Dense(mnistData.shape[1], activation='linear',
                             input_dim = mnistData.shape[1]))
model.add(keras.layers.Dense(10, activation='softmax'))
# Compile
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train
trainHistory = model.fit(mnistData, mnistLabels, epochs=10, batch_size=100,
                         validation_split=0.1, verbose=0)
# Plot
showClassificationResults(trainHistory)

###############################################

# Train a nonlinear model, better quality
model = None
# Define
model = keras.Sequential()
model.add(keras.layers.Dense(mnistData.shape[1], activation='relu', # use 'relu'
                             input_dim=mnistData.shape[1]))
model.add(keras.layers.Dense(10, activation='softmax'))
# Compile
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train
trainHistory = model.fit(mnistData, mnistLabels, epochs=20, batch_size=100,
                        validation_split=0.1, verbose=0)
# Plot
showClassificationResults(trainHistory)

###############################################

# Adding a second layer, minor improvement
model = None
# Define
model = keras.Sequential()
model.add(keras.layers.Dense(mnistData.shape[1], activation='relu',
                             input_dim = mnistData.shape[1]))
model.add(keras.layers.Dense(mnistData.shape[1], activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
# Compile
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train
trainHistory = model.fit(mnistData, mnistLabels, epochs=10, batch_size=100, validation_split=0.1, verbose=0)
# Plot
showClassificationResults(trainHistory)

# Check for training/validation data skew
# The distribution of classes in training and validation data is similar.
f = plt.figure(figsize=(10,3))
ax = f.add_subplot(1,2,1)
plt.hist(mnistLabels[0:round(len(mnistLabels)*8/10)], bins=range(numClasses+1))
plt.xticks(range(numClasses+1))
ax2 = f.add_subplot(1,2,2,)
plt.hist(mnistLabels[round(len(mnistLabels)*8/10):-1], bins=range(numClasses+1))
plt.xticks(range(numClasses+1))

###############################################

# Apply dropout regularization
# Try to reduce the divergence between training and validation loss
from keras import regularizers
model = None
# Define lambda
dropoutLambda = 0.5 #@param
# Define model
model = keras.Sequential()
model.add(keras.layers.Dense(mnistData.shape[1],
                             input_dim=mnistData.shape[1],
                             activation='relu'))
model.add(keras.layers.Dropout(dropoutLambda,
                               noise_shape=(1, mnistData.shape[1])))
model.add(keras.layers.Dense(10, activation='softmax'))
# Compile
model.compile(optimizer = "adam",
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
# Train
trainHistory = model.fit(mnistData,
                        mnistLabels,
                        epochs=30,
                        batch_size=500,
                        validation_split=0.1,
                        verbose=0)
# Plot
showClassificationResults(trainHistory)

# Check accuracy for data slices (different classes)
# The classification metrics are very uniform across all classes, which is perfect.
from sklearn.metrics import classification_report
mnistPred = np.argmax(model.predict(x = mnistData), axis=-1)
print(classification_report(mnistLabels, mnistPred))