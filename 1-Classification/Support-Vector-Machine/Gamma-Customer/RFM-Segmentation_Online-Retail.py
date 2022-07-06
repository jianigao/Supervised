import numpy as np
import pandas as pd

#import visualization tools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mat

df = pd.read_excel("../Datasets/Online_Retail.xlsx")
df.head()

df1 = df.copy()
df1.Country.nunique() # number of unique countries

df1.Country.unique()

customer_country=df1[['Country','CustomerID']].drop_duplicates()
customer_country.groupby(['Country'])['CustomerID'].aggregate('count').sort_values(ascending=False).reset_index()
# 3950 customers from the United Kingdom.

# Focus on the UK
df1 = df1.loc[df1['Country'] == 'United Kingdom']
# Remove empty CustomerID values
df1 = df1[pd.notnull(df1['CustomerID'])]
# Check if price, quantity > 0
df1.UnitPrice.min()
df1.Quantity.min()
df1 = df1[(df1['Quantity']>0)]

df1.shape
df1.info()

# Check unique value for each column.
def unique_counts(df1):
   for i in df1.columns:
       count = df1[i].nunique()
       print(i, ": ", count)
unique_counts(df1)

# Add a feature of TotalPrice
df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']

# Find out the first and last order dates in the data.
df1['InvoiceDate'].min()
df1['InvoiceDate'].max()

# Since recency is calculated for a point in time, 
# and the last invoice date is 2011–12–09, 
# we will use 2011–12–10 to calculate recency.

import datetime as dt
NOW = dt.datetime(2011,12,10)
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])

# RFM Customer Segmentation
rfmTable = df1.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (NOW - x.max()).days, 
    'InvoiceNo': lambda x: len(x), 
    'TotalPrice': lambda x: x.sum()})
rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.columns = ['recency','frequency','monetary']
rfmTable

# Split the metrics
quantiles = rfmTable.quantile(q=[0.25,0.75])
quantiles = quantiles.to_dict()

# get rank for each metrics
def rank_r(x, p, t):
    if x <= t[p][0.25]:
        return str(1)
    elif x <= t[p][0.75]: 
        return str(2)
    else:
        return str(3)
    
def rank_f(x, p, t):
    if x <= t[p][0.25]:
        return str(3)
    elif x <= t[p][0.75]: 
        return str(2)
    else:
        return str(1)
    
def rank_m(x, p, t):
    if x <= t[p][0.25]:
        return str(3)
    elif x <= t[p][0.75]: 
        return str(2)
    else:
        return str(1)

segmented_rfm = rfmTable
segmented_rfm['rank_r'] = segmented_rfm['recency'].apply(rank_r, args=('recency',quantiles))
segmented_rfm['rank_f'] = segmented_rfm['frequency'].apply(rank_f, args=('frequency',quantiles))
segmented_rfm['rank_m'] = segmented_rfm['monetary'].apply(rank_m, args=('monetary',quantiles))
segmented_rfm.head()

# Add a new column to combine RFM score: 
# 111 is the highest score as we determined earlier.
segmented_rfm['rfm_score'] = segmented_rfm.rank_r + segmented_rfm.rank_f + segmented_rfm.rank_m
segmented_rfm.head()

segmented_rfm.rfm_score.nunique()

df2 = segmented_rfm[segmented_rfm['rfm_score']=='111'].sort_values('monetary', ascending=False)
df2.reset_index(inplace=True)

df3 = df2['CustomerID']
df3.to_csv("../Datasets/BestCustomer.csv")

def define_rfm_segment(rows):
    if rows['rfm_score'] == '111':
        return 'best_users'
    elif rows['rfm_score'] == '211':
        return 'almost_lost'
    elif rows['rfm_score'] == '311':
        return 'lost_users'
    elif rows['rank_r'] == '3':
        return 'cheap_lost'
    elif rows['rank_f'] == '1':
        return 'loyal_users'
    elif rows['rank_m'] == '1':
        return 'big_spender'
    elif rows['rank_f'] == '3':
        return 'new_customer'
    else:
        return rows['rfm_score']    
    
#Define segment based on RFM score 
segmented_rfm['segment'] = segmented_rfm.apply(define_rfm_segment, axis =1)
segmented_rfm

segmented_rfm.segment.unique()

# Visualize the user segments
segmented_rfm.reset_index(inplace=True)
segmented_rfm_count = segmented_rfm.groupby('segment').agg({'CustomerID':['count'],'monetary':['sum']}).reset_index()
segmented_rfm_count.columns = ['segment','user','amount']
segmented_rfm_count[['amount']] = segmented_rfm_count[['amount']]/100
fig1, ax1 = plt.subplots()
ax1.pie(data=segmented_rfm_count, x='user', labels='segment',  autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Percentage of user in each segement")
plt.show()

# Visualize the amount spent by user segments
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(data=segmented_rfm_count, x='amount', labels='segment',  autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.25,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Percentage of amount spent by each group")
plt.show()

# function to label each user as high growth or not
def Label_Segments(rows):
    if rows['segment'] == 'best_users':
        return 1
    elif rows['segment'] == 'big_spender':
        return 1
    else:
        return 0

segmented_rfm['high_growth'] = segmented_rfm.apply(Label_Segments, axis =1)
user_label = segmented_rfm[['CustomerID', 'high_growth']]
user_label

# Visualize High Growth Users
growth_count=segmented_rfm.groupby('high_growth').agg({'CustomerID':['count'],'monetary':['sum']}).reset_index()
growth_count.columns = ['segment','user','amount']
growth_count.loc[growth_count['segment']==0, 'segment'] ='low growth'
growth_count.loc[growth_count['segment']==1, 'segment'] ='high growth'

fig1, ax1 = plt.subplots()
ax1.pie(data=growth_count, x='user', labels='segment',  autopct='%1.1f%%', startangle=90)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Percentage of user in each segement")
plt.show()

# Amount Spent by High Growth Users
fig1, ax1 = plt.subplots()
ax1.pie(data=growth_count, x='amount', labels='segment',  autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Amount spent by each group")
plt.show()
