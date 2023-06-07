# import methods as m
# import model as model
import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

data_files = ['measurements/Tek0000.csv', 'measurements/Tek0001.csv',
              'measurements/Tek0002.csv', 'measurements/Tek0003.csv',
              'measurements/Tek0004.csv', 'measurements/Tek0005.csv',
              'measurements/Tek0006.csv']
dfs = []
for file in data_files:
    df = pd.read_csv(file)
    dfs.append(df)

dataset = pd.concat(dfs)
okres = dataset
parametry1 = {
    'okres_t': okres['TIME'].max() - okres['TIME'].min(),
    'wartosc_srednia': okres['CH1'].mean(),
    'odchylenie': okres['CH1'].std(),
    'mediana': okres['CH1'].median(),
    'min': okres['CH1'].min(),
    'max': okres['CH1'].max(),
    'srednia_harmoniczna': len(okres) / np.sum(1 / okres['CH1']),
    'dynamika_zmian': okres['CH1'].max() - okres['CH1'].min()
}
parametry2 = {
    'okres_t': okres['TIME'].max() - okres['TIME'].min(),
    'wartosc_srednia': okres['CH2'].mean(),
    'odchylenie': okres['CH2'].std(),
    'mediana': okres['CH2'].median(),
    'min': okres['CH2'].min(),
    'max': okres['CH2'].max(),
    'srednia_harmoniczna': len(okres) / np.sum(1 / okres['CH2']),
    'dynamika_zmian': okres['CH2'].max() - okres['CH2'].min()
}
print(parametry1)

# Tworzenie ramki danych na podstawie parametrów
data = pd.DataFrame([parametry1, parametry2])

# Podział danych na cechy (X) i etykiety (y)
X = data.drop(['wartosc_srednia'], axis=1)  # Usuń kolumnę 'wartosc_srednia' jako etykietę
y = data['wartosc_srednia']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja i uczenie modelu sieci neuronowej
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train, y_train)

# Ocena modelu na zbiorze testowym
score = model.score(X_test, y_test)
print("R^2 score:", score)

# Przykładowe przewidywanie na nowych danych
new_data = pd.DataFrame([
    {'okres_t': 10, 'odchylenie': 0.5, 'mediana': 5, 'min': 0, 'max': 10, 'srednia_harmoniczna': 3, 'dynamika_zmian': 10}
])
prediction = model.predict(new_data)
print("Przewidywana wartość średnia:", prediction)


# old main -------------------------------------------------------------------------------------------------------------
# dataL = m.read_from_parquet('0kg_L')
# dataR = m.read_from_parquet('0kg_R')
# dataW = m.read_from_parquet('0kg_W')

# m.generate_graph(dataL, dataR, dataW)

# model.model_MLP(dataL, dataR, dataW)
# old main -------------------------------------------------------------------------------------------------------------
