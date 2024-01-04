# Importing the Libraries
import pandas as PD
import numpy as np
# matplotlib inline
import matplotlib. pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.plot_model import plot_model

#Get the Dataset
df=pd.read_csv("MicrosoftStockData.csv", na_values = ['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
df.head()

