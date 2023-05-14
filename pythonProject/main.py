import methods as m
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Wczytać dane, nadać etykiety do każdego wektora.
# Zastosować filtrację z przesuwnym oknem, np. medianową lub uśredniającą (doczytać sobie) z różną długością okien, np. 5, 9, 11, 15, 21. Po tej operacji będziecie mieć dla każdego pliku po kilka przefiltrowanych wektorów różnymi metodami (miejcie je wszystkie w pamięci).
# Ze względu na duże zbiory danych proponuję wykonać decymację sygnału (to wam skróci wektory wejściowe), np. co 5, 10, 50, 100 próbka.
# Wykreślić przebiegi parami (koło L/P) dla ww czynności.
# Podzielić każdy z przebiegów na stałe odcinki czasu, np. po 10ms, 20ms lub 50ms. 
# sprawdzić ile jest wartości w każdym z przedziałów (w zasadzie, chodzi o najdłuższy wektor). należy "uzupełnić" dane w każdym z wektorów tak, aby miały takie same długości, np. 500.
# Przygotowywać dokument, żeby był ślad po tym co robicie.

dataL = m.read_from_parquet('0kg_L')
dataR = m.read_from_parquet('0kg_R')
dataW = m.read_from_parquet('0kg_W')

dfa = pd.concat([dataW, dataL, dataR])
data = m.regression_with_preprocessed_data(dfa)
# Usunięcie wierszy zawierających brakujące wartości
# df.dropna(inplace=True)
# Podział danych na cechy (X) i etykiety (y)
df = data['df_preprocessed']
X = df['Czas'].values.reshape(-1, 1)
y = df['CH1_filtered'].values

# Imputacja brakujących wartości w danych X
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)
# Usunięcie wierszy zawierających brakujące wartości
non_nan_indices = ~np.isnan(X_imputed[:, 0])  # Indeksowanie tylko pierwszej kolumny
X_cleaned = X_imputed[non_nan_indices]
y_cleaned = y[non_nan_indices]

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standaryzacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicjalizacja modelu MLP
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42)

# Trenowanie modelu
model.fit(X_train_scaled, y_train)

# Predykcja na danych testowych
y_pred = model.predict(X_test_scaled)

# Obliczenie średniego błędu kwadratowego
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

graphL = m.regression_with_preprocessed_data(dataL)
graphR = m.regression_with_preprocessed_data(dataR)
graphW = m.regression_with_preprocessed_data(dataW)

df_preprocessedL = graphL['df_preprocessed']
regression_lineL = graphL['regression_line']
df_preprocessedR = graphR['df_preprocessed']
regression_lineR = graphR['regression_line']
df_preprocessedW = graphW['df_preprocessed']
regression_lineW = graphW['regression_line']

plt.figure("Graph")
plt.scatter(df_preprocessedL["Czas"], df_preprocessedL["CH1_filtered"], label='0kgL', color='purple')
plt.plot(df_preprocessedL["Czas"], regression_lineL, color='purple', label='0kgL reg.')
plt.scatter(df_preprocessedR["Czas"], df_preprocessedR["CH1_filtered"], label='0kgR', color='blue')
plt.plot(df_preprocessedR["Czas"], regression_lineR, color='blue', label='0kgR reg.')
plt.scatter(df_preprocessedW["Czas"], df_preprocessedW["CH1_filtered"], label='0kgW', color='green')
plt.plot(df_preprocessedW["Czas"], regression_lineW, color='green', label='0kgW reg.')

plt.title("Graph showing the dependence of Time on CH1")
plt.xlabel("Time")
plt.ylabel("CH1")
plt.legend()
plt.show()

