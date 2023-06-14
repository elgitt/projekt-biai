import methods as m
# import model as model
import pandas as pd
import numpy as np
import csv
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPClassifier

data = m.get_vectors()

vec0 = data['vec0']
vec1 = data['vec1']
vec2 = data['vec2']
vec3 = data['vec3']
vec4 = data['vec4']
vec5 = data['vec5']

# old main -------------------------------------------------------------------------------------------------------------
# dataL = m.read_from_parquet('0kg_L')
# dataR = m.read_from_parquet('0kg_R')
# dataW = m.read_from_parquet('0kg_W')

# m.generate_graph(dataL, dataR, dataW)

# model.model_MLP(dataL, dataR, dataW)
# old main -------------------------------------------------------------------------------------------------------------
