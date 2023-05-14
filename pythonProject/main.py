import methods as m
import matplotlib.pyplot as plt

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

graphL = m.regression_with_preprocessed_data(dataL)
graphR = m.regression_with_preprocessed_data(dataR)
graphW = m.regression_with_preprocessed_data(dataW)

df_preprocessedL = graphL['df_preprocessed']
regression_lineL = graphL['regression_line']
df_preprocessedR = graphL['df_preprocessed']
regression_lineR = graphL['regression_line']
df_preprocessedW = graphL['df_preprocessed']
regression_lineW = graphL['regression_line']

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

