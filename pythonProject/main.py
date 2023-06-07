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
print(parametry2)



# old main -------------------------------------------------------------------------------------------------------------
# dataL = m.read_from_parquet('0kg_L')
# dataR = m.read_from_parquet('0kg_R')
# dataW = m.read_from_parquet('0kg_W')

# m.generate_graph(dataL, dataR, dataW)

# model.model_MLP(dataL, dataR, dataW)
# old main -------------------------------------------------------------------------------------------------------------
