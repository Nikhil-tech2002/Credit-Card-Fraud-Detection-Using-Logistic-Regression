#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt



# In[27]:


df = pd.read_csv("fraudTrain.csv")   


# In[28]:


df = df[df['amt'] > 50].reset_index(drop=True)


# In[29]:


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour


# In[30]:


df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365


# In[31]:


df.drop(columns=[
    'cc_num', 'first', 'last', 'street', 'city',
    'state', 'zip', 'trans_num', 'merchant',
    'dob', 'trans_date_trans_time'
], inplace=True)


# In[32]:


num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna('Unknown')


# In[33]:


for col in ['category', 'gender', 'job']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# In[34]:


TARGET = 'is_fraud'
X = df.drop(TARGET, axis=1)
y = df[TARGET]


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[37]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[38]:


model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)


# In[39]:


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# In[41]:


plt.figure(figsize=(6,4))
plt.scatter(df['long'], df['lat'], c=df['is_fraud'], alpha=0.3)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Transaction Locations (Fraud Highlighted)")
plt.show()

