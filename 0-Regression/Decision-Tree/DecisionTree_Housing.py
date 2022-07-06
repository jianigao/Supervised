# type() (what is this thing?)
# dir() (what can I do with it?)
# help() (tell me more)

import pandas as pd # DataFrame
from sklearn.tree import DecisionTreeRegressor # To create model
from sklearn.metrics import mean_absolute_error # Model evaluation/validation
from sklearn.model_selection import train_test_split

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Print a summary of the data in Melbourne data
melbourne_data.describe()
# The first number, the count, shows how many rows have non-missing values.
# To see a list of all columns in the dataset
melbourne_data.columns
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
# Select the prediction target y
y = melbourne_data.Price
# Choose features X to be inputted into model to make predictions
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
# Review data
X.describe()
# Print the top few lines
X.head()

# Split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    # Define model
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    # Fit model
    model.fit(train_X, train_y)
    # Get predictions on validation data
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores, key=scores.get)
# Fill in argument to make optimal size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
# Fit the final model with all the data to be more accurate
# No need to hold out the valid data since all the modeling decisions are made
final_model.fit(X, y)
