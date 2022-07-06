from os.path import join # for joining file pathnames
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Set Pandas display options
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

wineDf = pd.read_csv(
  "https://download.mlcc.google.com/mledu-datasets/winequality.csv",
  encoding='latin-1')
wineDf.columns = ['fixed acidity','volatile acidity','citric acid',
                     'residual sugar','chlorides','free sulfur dioxide',
                     'total sulfur dioxide','density','pH',
                     'sulphates','alcohol','quality']
wineDf.head()

# Check correlation matrix
corr_wineDf = wineDf.corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr_wineDf, annot=True)

# Define function to validate input data against data schema
def test_data_schema(input_data, schema):
  """Tests that the datatypes and ranges of values in the dataset
    adhere to expectations.

    Args:
      input_function: Dataframe containing data to test
      schema: Schema which describes the properties of the data.
  """

  def test_dtypes():
    for column in schema.keys():
      assert input_data[column].map(type).eq(
          schema[column]['dtype']).all(), (
          "Incorrect dtype in column '%s'." % column
      )
    print('Input dtypes are correct.')

  def test_ranges():
    for column in schema.keys():
      schema_max = schema[column]['range']['max']
      schema_min = schema[column]['range']['min']
      # Assert that data falls between schema min and max.
      assert input_data[column].max() <= schema_max, (
          "Maximum value of column '%s' is too low." % column
      )
      assert input_data[column].min() >= schema_min, (
          "Minimum value of column '%s' is too high." % column
      )
    print('Data falls within specified ranges.')

  test_dtypes()
  test_ranges()

wineDf.describe()

wine_schema = {
    'fixed acidity': {
        'range': {
            'min': 3.7,
            'max': 15.9
        },
        'dtype': float,
    },
    'volatile acidity': {
        'range': {
            'min': 0.08,  # minimum value
            'max': 1.6   # maximum value
        },
        'dtype': float,    # data type
    },
    'citric acid': {
        'range': {
            'min': 0.0, # minimum value
            'max': 1.7  # maximum value
        },
        'dtype': float,   # data type
    }
}

print('Validating wine data against data schema...')
test_data_schema(wineDf, wine_schema)

# Shuffle data
wineDf = wineDf.sample(frac=1).reset_index(drop=True)
# Split the dataset into data and labels
wineFeatures = wineDf.copy(deep=True)
wineFeatures.drop(columns='quality',inplace=True)
wineLabels = wineDf['quality'].copy(deep=True)

# Normalize data using z-score
def normalizeData(arr):
  stdArr = np.std(arr)
  meanArr = np.mean(arr)
  arr = (arr-meanArr)/stdArr
  return arr

for str1 in wineFeatures.columns:
   wineFeatures[str1] = normalizeData(wineFeatures[str1])

# Set up testing functions
import unittest

def test_input_dim(df, n_rows, n_columns):
  assert len(df) == n_rows, "Unexpected number of rows."
  assert len(df.columns) == n_columns, "Unexpected number of columns."
  print('Engineered data has the expected number of rows and columns.')

def test_nulls(df):
  dataNulls = df.isnull().sum().sum()
  assert dataNulls == 0, "Nulls in engineered data."
  print('Engineered features do not contain nulls.')

# Test dimensions of engineered data
wine_feature_rows = 6497 #@param
wine_feature_cols = 11 #@param
test_input_dim(wineFeatures,
               wine_feature_rows,
               wine_feature_cols)

test_nulls(wineFeatures)

# Split training and validation sets, both sets should be similar.
splitIdx = round(wineFeatures.shape[0]*8/10)
wineFeatures.iloc[0:splitIdx,:].describe()
wineFeatures.iloc[splitIdx:-1,:].describe()

# Establish a baseline
# For a regression problem, the simplest baseline to predict the average value. 
baselineMSE = np.square(wineLabels[0:splitIdx]-np.mean(wineLabels[0:splitIdx]))
baselineMSE = np.sum(baselineMSE)/len(baselineMSE)
print(baselineMSE)

def showRegressionResults(trainHistory):
  """Function to:
   * Print final loss.
   * Plot loss curves.
  
  Args:
    trainHistory: object returned by model.fit
  """
  
  # Print final loss
  print("Final training loss: " + str(trainHistory.history['loss'][-1]))
  print("Final Validation loss: " + str(trainHistory.history['val_loss'][-1]))
  
  # Plot loss curves
  plt.plot(trainHistory.history['loss'])
  plt.plot(trainHistory.history['val_loss'])
  plt.legend(['Training loss','Validation loss'],loc='best')
  plt.title('Loss Curves')

###############################################

# Define a linear model, resulting loss < baseline loss
model = None
# Choose feature
wineFeaturesSimple = wineFeatures['alcohol']
# Define model
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, activation='linear', input_dim=1))
# Specify the optimizer using the TF API to specify the learning rate
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
              loss='mse')
# Train the model!
trainHistory = model.fit(wineFeaturesSimple,
                         wineLabels,
                         epochs=20,
                         batch_size=100, # set batch size here
                         validation_split=0.2,
                         verbose=0)
# Plot
showRegressionResults(trainHistory)

###############################################

# Add feature to linear model, a smaller loss
model = None
# Select features
wineFeaturesSimple = wineFeatures[['alcohol', 'volatile acidity']] # add second feature
# Define model
model = keras.Sequential()
model.add(keras.layers.Dense(wineFeaturesSimple.shape[1],
                             input_dim=wineFeaturesSimple.shape[1],
                             activation='linear'))
model.add(keras.layers.Dense(1, activation='linear')) # add second layer
# Compile
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mse')
# Train
trainHistory = model.fit(wineFeaturesSimple,
                         wineLabels,
                         epochs=20,
                         batch_size=100,
                         validation_split=0.2,
                         verbose=0)
# Plot results
showRegressionResults(trainHistory)

###############################################

# Use a nonlinear model, minor improvement
model = None
# Define
model = keras.Sequential()
model.add(keras.layers.Dense(wineFeaturesSimple.shape[1],
                             input_dim=wineFeaturesSimple.shape[1],
                             activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
# Compile
model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
# Fit
model.fit(wineFeaturesSimple,
          wineLabels,
          epochs=20,
          batch_size=100,
          validation_split=0.2,
          verbose=0)
# Plot results
showRegressionResults(trainHistory)

###############################################

# Optimize your model by using more features, minor improvement
# Choose features
wineFeaturesSimple = wineFeatures[['alcohol','volatile acidity','chlorides','density']]
# Define
model = None
model = keras.Sequential()
model.add(keras.layers.Dense(wineFeaturesSimple.shape[1],
                             activation='relu',
                             input_dim=wineFeaturesSimple.shape[1]))
# Add more layers here
model.add(keras.layers.Dense(1,activation='linear'))
# Compile
model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
# Train
trainHistory = model.fit(wineFeaturesSimple,
                         wineLabels,
                         epochs=200,
                         batch_size=100,
                         validation_split=0.2,
                         verbose=0)
# Plot results
showRegressionResults(trainHistory)

###############################################

# Check for implementation bugs using reduced dataset
# You get a low loss on your reduced dataset. This result means your model is 
# probably solid and your previous results are as good as they'll get.
# Choose 10 examples
wineFeaturesSmall = wineFeatures[0:10]
wineLabelsSmall = wineLabels[0:10]
# Define model
model = None
model = keras.Sequential()
model.add(keras.layers.Dense(wineFeaturesSmall.shape[1], activation='relu',
                             input_dim=wineFeaturesSmall.shape[1]))
model.add(keras.layers.Dense(wineFeaturesSmall.shape[1], activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
# Compile
model.compile(optimizer=tf.optimizers.Adam(0.01), loss='mse') # set LR
# Train
trainHistory = model.fit(wineFeaturesSmall,
                        wineLabelsSmall,
                        epochs=200,
                        batch_size=10,
                        verbose=0)
# Plot results
print("Final training loss: " + str(trainHistory.history['loss'][-1]))
plt.plot(trainHistory.history['loss'])

###############################################

# Try a very complex model with all the features
# The minor loss decrease isn't worth the complexity.
# However, it dependes on the objectives.
model = None
# Define
model = keras.Sequential()
model.add(keras.layers.Dense(wineFeatures.shape[1], activation='relu',
                             input_dim=wineFeatures.shape[1]))
model.add(keras.layers.Dense(wineFeatures.shape[1], activation='relu'))
model.add(keras.layers.Dense(wineFeatures.shape[1], activation='relu'))
model.add(keras.layers.Dense(1,activation='linear'))
# Compile
model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
# Train the model!
trainHistory = model.fit(wineFeatures, wineLabels, epochs=100, batch_size=100,
                         verbose=1, validation_split = 0.2)
# Plot results
showRegressionResults(trainHistory)
plt.ylim(0.4,1)