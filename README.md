# LSTM-Stock-Prediction
Stock price prediction using the LSTM machine learning network implemented via Tensor Flow Keras and Python.  
USE THIS AT YOUR OWN RISK! I am not responsible for any loss caused by using this program, and it shall not help you make decisions on trading stock. This project is only a proof-of-concept.  
  
Required libraries:  
from tensorflow.keras.layers import Dense, LSTM, Dropout  
from tensorflow.keras import Sequential, initializers  
import tensorflow as tf  
from sklearn.preprocessing import MinMaxScaler  
from pandas_datareader import data  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import datetime as datetime  
import sys  
import os  
  
Required environment:   
Nvidia CUDA Tool kit 10.1  
Nvidia CuDNN.dll lastest version built for CUDA 10.1  
Python 3.7  
  
The model trained for this project is so far only applicable to Nvidia Stock (NASDAQ: NVDA). If you would like to use it for other stocks, you have to build and train your own model, as each stock requires a different model.  
  
LSTM Stock Prediction V2_9.py is a complete set of features, from building, training models, to the actual prediction part  
predict.py is only able to load a trained model and make predictions according to data obtained.  
test model.py is for testing model accuracy only.  
