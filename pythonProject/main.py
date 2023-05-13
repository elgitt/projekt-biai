import methods as m
import matplotlib.pyplot as plt
import numpy as np

# Wczytać dane, nadać etykiety do każdego wektora.
# Zastosować filtrację z przesuwnym oknem, np. medianową lub uśredniającą (doczytać sobie) z różną długością okien, np. 5, 9, 11, 15, 21. Po tej operacji będziecie mieć dla każdego pliku po kilka przefiltrowanych wektorów różnymi metodami (miejcie je wszystkie w pamięci).
# Ze względu na duże zbiory danych proponuję wykonać decymację sygnału (to wam skróci wektory wejściowe), np. co 5, 10, 50, 100 próbka.
# Wykreślić przebiegi parami (koło L/P) dla ww czynności.
# Podzielić każdy z przebiegów na stałe odcinki czasu, np. po 10ms, 20ms lub 50ms. 
# sprawdzić ile jest wartości w każdym z przedziałów (w zasadzie, chodzi o najdłuższy wektor). należy "uzupełnić" dane w każdym z wektorów tak, aby miały takie same długości, np. 500.
# Przygotowywać dokument, żeby był ślad po tym co robicie.

test = m.read_from_parquet('0kg_L')

df_preprocessed = m.preprocess_data(test, decimation_step=10, filter_window_size=20, time_step=2)
print(df_preprocessed)
m.regression_model(df_preprocessed)

coefficients = np.polyfit(df_preprocessed["Czas"], df_preprocessed["CH1_filtered"], 1)
slope = coefficients[0]
intercept = coefficients[1]
regression_line = slope * df_preprocessed["Czas"] + intercept

plt.scatter(df_preprocessed["Czas"], df_preprocessed["CH1_filtered"], label='L', color='blue')
plt.plot(df_preprocessed["Czas"], regression_line, color='purple', label='Regression')
plt.title("Przebiegi parami")
plt.xlabel("Czas")
plt.ylabel("Wartości pomiarów")
plt.legend()
plt.show()

