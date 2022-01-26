from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import os
import pandas as pd
import time

def predict_weather():
    data = get_data()
    data = process_data(data)
    x_train, y_train, x_test, y_test = get_data_train_test(data)
    linear_regression = train_model(x_train, y_train)
    prediction = pd.DataFrame(linear_regression.predict(x_test))

def get_data():
    files = os.listdir('weather-data/')
    data_frames = [pd.read_csv('weather-data/' + filename) for filename in files]
    data = pd.concat(data_frames).reset_index()
    
    return data

def process_data(data):
    data = data.drop([
        'Date', 'index', 'Longitude', 'Latitude', 'Elevation', 'Solar'
    ], axis = 1)
    
    return data

def get_data_train_test(data):
    data_train, data_test = train_test_split(data, test_size = 0.3, random_state = 0)
    x_train = data_train[['Max Temperature', 'Precipitation', 'Wind', 'Relative Humidity']] # Independent variables
    y_train = data_train[['Min Temperature']] # Dependent variable
    x_test = data_test[['Max Temperature', 'Precipitation', 'Wind', 'Relative Humidity']]
    y_test = data_test[['Min Temperature']]

    return x_train, y_train, x_test, y_test

def train_model(x_train, y_train):
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(x_train, y_train)

    return linear_regression

if __name__ == '__main__':
    start = time.time()
    predict_weather()
    end = time.time()

    print("Tiempo de ejecución: " + str(end - start))