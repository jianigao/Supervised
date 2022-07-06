#!/usr/bin/env python
# coding: utf-8

# # Introduction #
# 
# Run this cell to set everything up!

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

def make_lags(ts, lags, lead_time=1, name='y'):
    return pd.concat(
        {
            f'{name}_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)

def make_multistep_target(ts, steps, reverse=False):
    shifts = reversed(range(steps)) if reverse else range(steps)
    return pd.concat({f'y_step_{i + 1}': ts.shift(-i) for i in shifts}, axis=1)

def create_multistep_example(n, steps, lags, lead_time=1):
    ts = pd.Series(
        np.arange(n),
        index=pd.period_range(start='2010', freq='A', periods=n, name='Year'),
        dtype=pd.Int8Dtype,
    )
    X = make_lags(ts, lags, lead_time)
    y = make_multistep_target(ts, steps, reverse=True)
    data = pd.concat({'Targets': y, 'Features': X}, axis=1)
    data = data.style.set_properties(['Targets'], **{'background-color': 'LavenderBlush'}) \
                     .set_properties(['Features'], **{'background-color': 'Lavender'})
    return 

def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return 


# In[2]:


store_sales = pd.read_csv(
    'Datasets/train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)

test = pd.read_csv(
    'Datasets/test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['store_nbr', 'family', 'date']).sort_index()


# -------------------------------------------------------------------------------
# 
# # 1) Identify the forecasting task for *Store Sales* competition
# 
# Look at the time indexes of the training and test sets. From this information, can you identify the forecasting task for *Store Sales*?

# In[3]:


print("Training Data", "\n" + "-" * 13 + "\n", store_sales)
print("\n")
print("Test Data", "\n" + "-" * 9 + "\n", test)


# Try to identify the *forecast origin* and the *forecast horizon*. How many steps are within the forecast horizon? What is the lead time for the forecast?

# The training set ends on `2017-08-15`, which gives us the forecast origin. The test set comprises the dates `2017-08-16` to `2017-08-31`, and this gives us the forecast horizon. There is one step between the origin and horizon, so we have a lead time of one day.
# 
# Put another way, we need a 16-step forecast with a 1-step lead time. We can use lags starting with lag 1, and we make the entire 16-step forecast using features from `2017-08-15`.

# -------------------------------------------------------------------------------
# 
# In the tutorial we saw how to create a multistep dataset for a single time series. Fortunately, we can use exactly the same procedure for datasets of multiple series.
# 
# # 2) Create multistep dataset for *Store Sales*
# 
# Create targets suitable for the *Store Sales* forecasting task. Use 4 days of lag features. Drop any missing values from both targets and features.

# In[4]:


y = family_sales.loc[:, 'sales']

# Make 4 lag features
X = make_lags(y, lags=4).dropna()

# Make multistep target
y = make_multistep_target(y, steps=16).dropna()

y, X = y.align(X, join='inner', axis=0)


# -------------------------------------------------------------------------------
# 
# In the tutorial, we saw how to forecast with the MultiOutput and Direct strategies on the *Flu Trends* series. Now, you'll apply the DirRec strategy to the multiple time series of *Store Sales*.
# 
# Make sure you've successfully completed the previous exercise and then run this cell to prepare the data for XGBoost.

# In[5]:


le = LabelEncoder()
X = (X
    .stack('family')  # wide to long
    .reset_index('family')  # convert index to column
    .assign(family=lambda x: le.fit_transform(x.family))  # label encode
)
display(X)


# In[6]:


y = y.stack('family')  # wide to long
display(y)


# # 3) Forecast with the DirRec strategy
# 
# Instatiate a model that applies the DirRec strategy to XGBoost.

# In[7]:


from sklearn.multioutput import RegressorChain

model = RegressorChain(base_estimator=XGBRegressor())


# Run this cell if you'd like to train this model.

# In[8]:


model.fit(X, y)

y_pred = pd.DataFrame(
    model.predict(X),
    index=y.index,
    columns=y.columns,
).clip(0.0)


# And use this code to see a sample of the 16-step predictions this model makes on the training data.

# In[9]:


FAMILY = 'BEAUTY'
START = '2017-04-01'
EVERY = 16

y_pred_ = y_pred.xs(FAMILY, level='family', axis=0).loc[START:]
y_ = family_sales.loc[START:, 'sales'].loc[:, FAMILY]

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax = y_.plot(**plot_params, ax=ax, alpha=0.5)
ax = plot_multistep(y_pred_, ax=ax, every=EVERY)
plt.legend([FAMILY, FAMILY + ' Forecast'])

