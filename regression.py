#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    response = requests.get(TRAIN_DATA_URL)
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

    # PREDICTION
    X1 = area.T
    y1 = lm.predict(X1)
    
    return y1.T

if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")

