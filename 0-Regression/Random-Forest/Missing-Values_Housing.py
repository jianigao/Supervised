import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Drop or impute missing values
##########################################################
# First Approach: Drop columns in training and validation data
new_X_train = X_train.drop(cols_with_missing, axis=1)
new_X_valid = X_valid.drop(cols_with_missing, axis=1)

##########################################################
# A Better Approach: Imputation

## Optional: Plus version ##
# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train[col + '_was_missing'] = X_train[col].isnull()
    X_valid[col + '_was_missing'] = X_valid[col].isnull()
## Optional: Plus version ##

my_imputer = SimpleImputer()
# my_imputer = SimpleImputer(strategy='median')
new_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
new_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
new_X_train.columns = X_train.columns
new_X_valid.columns = X_valid.columns
##########################################################

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

print("MAE:")
print(score_dataset(new_X_train, new_X_valid, y_train, y_valid))

# Preprocess test data
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

# Get test predictions
preds_test = model.predict(imputed_X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
