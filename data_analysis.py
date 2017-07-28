# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:39:14 2017

@author: prave
"""

import pandas as pd 
import datetime as dt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


scorer = make_scorer(mean_squared_error, greater_is_better = False)
test_data = pd.read_csv('test.csv') 
train_data = pd.read_csv('train.csv')

def change_date_type(data,col):
    
    data[col] = pd.to_datetime(data[col])
    data[col].astype('int64')
    data[col] = (data[col] - \
                                dt.datetime(1970,1,1)).dt.total_seconds()
    
    return data
    
    

def clean_data(data):
    
    """
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    
    data['pickup_datetime'].astype('int64')
    
    data['pickup_datetime'] = (data['pickup_datetime'] - \
                                dt.datetime(1970,1,1)).dt.total_seconds()
    """
    data = change_date_type(data,'pickup_datetime')
    data = change_date_type(data,'dropoff_datetime')
    data.loc[data['store_and_fwd_flag'] == 'Y','store_and_fwd_flag'] = 1
    data.loc[data['store_and_fwd_flag'] == 'N','store_and_fwd_flag'] = 0
    return data


def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, x_test, y_test, scoring = scorer, cv = 10))
    return(rmse)

def analysis_based_pickuptime(data):
    
    pass
    
    #data['pickup_time'] = pd.to_datetime(data['pickup_datetime'])
    #data['hour']   = data['pickup_time'].hour
    #print "PICK UP HOUR"
    #print data['pickup_hour'].unique()


if __name__ == "__main__":
    
    print train_data.columns
    print train_data.head()
    #analysis_based_pickuptime(train_data)
    
    
    train_data = clean_data(train_data)
    
    print train_data.columns
    
    print train_data.store_and_fwd_flag.unique()
    
    print train_data.dtypes
    
    y = train_data['trip_duration']
    x_train,x_test,y_train,y_test = train_test_split(train_data.ix[:, train_data.columns != 'id'],y, test_size=0.3,random_state=0)
    
    
    
    
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    print "RMSE on train set:",rmse_cv_train(lr).mean()
    print "RMSE on train set:",rmse_cv_test(lr).mean()
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc = "upper left")
    plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
    plt.show()
    
    # Plot predictions
    plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
    plt.show()
    