#!/usr/bin/env python
# coding: utf-8

# In[56]:


import requests
import pandas as pd
import scipy
import numpy as np
import sys


# In[76]:


from sklearn.linear_model import LinearRegression

# TRAINING MODEL

df = pd.read_csv("https://storage.googleapis.com/kubric-hiring/linreg_train.csv")
df_ = df.T
df_.reset_index(inplace = True)
X_ = df_.iloc[:,0]
y_ = df_.iloc[:,1]
X = np.array([X_[1:]])
y=np.array([y_[1:]])
lm = LinearRegression()
lm.fit(X,y)

# TESTING MODEL

df1 = pd.read_csv("https://storage.googleapis.com/kubric-hiring/linreg_test.csv")
df1_ = df1.T
df1_.reset_index(inplace = True)
X1_ = df1_.iloc[:,0]
X1 = np.array([X_[1:]])
y1 = lm.predict(X1)

