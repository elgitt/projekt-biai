# import methods as m
# import model as model
import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# wczytuje dane do jednego datasetu, pozdrawiam Ewunie
data_files = ['measurements/Tek0000.csv', 'measurements/Tek0001.csv',
              'measurements/Tek0002.csv', 'measurements/Tek0003.csv',
              'measurements/Tek0004.csv', 'measurements/Tek0005.csv',
              'measurements/Tek0006.csv']
dfs = []
for file in data_files:
    df = pd.read_csv(file)
    dfs.append(df)

dataset = pd.concat(dfs)
single_time = dataset['TIME']
print(dataset.head())

average = np.mean(dataset['CH1'])
std_deviation = np.std(dataset['CH1'])
median = np.median(dataset['CH1'])
minimum = np.min(dataset['CH1'])
maximum = np.max(dataset['CH1'])
fft_data = fft(dataset['CH1'].values)
dynamics = np.max(np.abs(fft_data))


print("Wyniki analizy:")
print("--------------")
print("Średnia wartość: ", average)
print("Odchylenie standardowe: ", std_deviation)
print("Mediana: ", median)
print("Wartość minimalna: ", minimum)
print("Wartość maksymalna: ", maximum)
print("Dynamika zmian FFT: ", dynamics)

plt.plot(dataset['TIME'], dataset['CH1'])
plt.xlabel('Czas')
plt.ylabel('Wartość CH1')
plt.title('Przebieg czasowy CH1')
plt.show()
plt.plot(dataset['TIME'], dataset['CH2'])
plt.xlabel('Czas')
plt.ylabel('Wartość CH2')
plt.title('Przebieg czasowy CH2')
plt.show()
# Przygotowanie danych
data = np.array([[average, std_deviation, median, minimum, maximum, dynamics]])

# Normalizacja danych (opcjonalnie)
# Przykład normalizacji do zakresu 0-1
# data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Definicja modelu sieci neuronowej
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(6)
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='mse')

# Trenowanie modelu
model.fit(data, data, epochs=100, batch_size=1)

# Testowanie modelu na nowych danych
predicted_data = model.predict(data)

print("Oszacowane dane:")
print("Średnia wartość: ", predicted_data[0][0])
print("Odchylenie standardowe: ", predicted_data[0][1])
print("Mediana: ", predicted_data[0][2])
print("Wartość minimalna: ", predicted_data[0][3])
print("Wartość maksymalna: ", predicted_data[0][4])
print("Dynamika zmian FFT: ", predicted_data[0][5])

# old main -------------------------------------------------------------------------------------------------------------
# dataL = m.read_from_parquet('0kg_L')
# dataR = m.read_from_parquet('0kg_R')
# dataW = m.read_from_parquet('0kg_W')

# m.generate_graph(dataL, dataR, dataW)

# model.model_MLP(dataL, dataR, dataW)
# old main -------------------------------------------------------------------------------------------------------------
