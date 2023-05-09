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

# m.return_median_mx('0kg_L')['parquet']
#test2 = m.return_median_mx('0kg_R')['parquet']

#decimated_vec = m.decimate_method(test, 100)
df_preprocessed = m.preprocess_data(test, decimation_step=10, filter_window_size=20, time_step=2)
print(df_preprocessed)
m.regression_model(df_preprocessed)
#decimated_vec2 = m.decimate_method(test2, 100)

#x1, y1 = m.change_to_vector(decimated_vec)
#x2, y2 = m.change_to_vector(decimated_vec2)

#test = decimated_vec.head(-1)
#test2 = decimated_vec2.head(-1)

plt.scatter(df_preprocessed["Czas"], df_preprocessed["CH1_filtered"], label='L')
#plt.scatter(test2["Czas"], test2["CH1"], label='P')
#plt.plot(x2, y2, 'r.', label='P')
plt.title("Przebiegi parami")
plt.xlabel("Czas")
plt.ylabel("Wartości pomiarów")
#plt.ylim(0, 1)
#plt.xlim(-2, 6)
#plt.yticks(np.arange(0, 0.1, 100))
plt.legend()
plt.show()


#m.regression_model(test)
# wektor z wektorow
# vectors = []
# for i in range(len(x1)):
#   vec = [x1[i], y1[i]]
#  vectors.append(vec)
#  print(vec)


# print("Original data shape: ", test.shape)
# print("Decimated data shape: ", decimated_vec.shape)
# print(decimated_vec)
