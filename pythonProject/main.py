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
from model import create_model
from tensorflow.keras.models import load_model

data = m.get_vectors()

vec0_data = np.array(data['vec0'])
vec1_data = np.array(data['vec1'])
vec2_data = np.array(data['vec2'])
vec3_data = np.array(data['vec3'])
vec4_data = np.array(data['vec4'])
vec5_data = np.array(data['vec5'])

print("vec0:" ,vec0_data)
print("vec1:" , vec1_data)
print("vec2:" , vec2_data)
print("vec3:" , vec3_data)
print("vec4:" , vec4_data)
print("vec5:" , vec5_data)

vec_test_data =np.array(data['vec3']);

# model = create_model()

# # Trenowanie modelu
# epochs = 10  # Przykładowa liczba epok
# batch_size = 32  # Przykładowy rozmiar paczki
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(vec0_data, np.zeros(len(vec0_data)), epochs=epochs, batch_size=batch_size)
# model.fit(vec1_data, np.ones(len(vec1_data)), epochs=epochs, batch_size=batch_size)
# model.fit(vec2_data, np.ones(len(vec2_data)) * 2, epochs=epochs, batch_size=batch_size)
# model.fit(vec3_data, np.ones(len(vec3_data)) * 3, epochs=epochs, batch_size=batch_size)
# model.fit(vec4_data, np.ones(len(vec4_data)) * 4, epochs=epochs, batch_size=batch_size)
# model.fit(vec5_data, np.ones(len(vec5_data)) * 5, epochs=epochs, batch_size=batch_size)
#
# prediction = model.predict(np.array([vec_test_data]))
# predicted_class = np.argmax(prediction)
#
# print("Predicted class:", predicted_class)
# old main -------------------------------------------------------------------------------------------------------------
# dataL = m.read_from_parquet('0kg_L')
# dataR = m.read_from_parquet('0kg_R')
# dataW = m.read_from_parquet('0kg_W')

# m.generate_graph(dataL, dataR, dataW)

# model.model_MLP(dataL, dataR, dataW)
# old main -------------------------------------------------------------------------------------------------------------
