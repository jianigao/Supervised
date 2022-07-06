import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load datasets
transaction_data = pd.read_csv("../Datasets/transaction_data.csv")
product_data = pd.read_csv("../Datasets/product_data.csv")

tran = transaction_data.copy()
prod = product_data.copy()

# dropna drops missing values
tran = tran.dropna(axis=0)
prod = prod.dropna(axis=0)

# Group the products by total purchased quantity
df1 = tran.groupby(['product_id'])['purchased_quantity'].sum().sort_values(ascending=False).reset_index()

# Find the most purchased product
pr = df1.product_id.iloc[0]

# Find the color of the most purchased product
cr = prod.loc[prod.product_id==pr, 'color'].unique()

# Convert the transaction date to datetime
tran['transaction_date'] = pd.to_datetime(tran['transaction_date'], format='%Y-%m-%d')

# Filter the data
df2 = tran[tran['transaction_date'] < '2012-10-01'].copy()

# Create new features
df2['money_spent'] = df2['purchased_quantity'] * df2['unit_product_price']

# Find the product id having color 'b'
a_pr = prod.loc[prod.color=='b', 'product_id'].unique()
a_pr

# If customer bought a product 'a'
df2['product'] = df2['product_id'].apply(lambda k: 1 if k==a_pr else 0)

# Aggregate data by customer
df3 = df2.groupby('customer_id').agg({
'transaction_id': lambda n: len(n),
'money_spent': lambda m: m.sum(),
'product': lambda k: k.max()})

# Change the name of columns
df3.columns = ['number_transactions','total_money_spent','bought_product_b']

# Add response feature
df3['response'] = tran.groupby('customer_id').agg({'transaction_date': lambda x: x.max()})
df3['response'] = (df3['response'] >= '2012-10-01').astype(int)

df3.to_csv("../Datasets/feature_data.csv")

X = df3.copy()
y = X.pop('response')

X = StandardScaler().fit_transform(X)

# Split data into training and validation data, for both features and target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state=0)

# SVC model
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20)
clf.fit(X_train, y_train)

score = clf.score(X_train, y_train)
print("Score: ", score)

y_pred = clf.predict(X_valid)

cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True)

cr = classification_report(y_valid, y_pred)
print(cr)
