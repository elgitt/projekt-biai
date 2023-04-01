import methods as m
import matplotlib.pyplot as plt
import numpy as np
import math


test = m.read_from_parquet('test')
m5 = m.median_method(test, 5)
m9 = m.median_method(test, 9)
m11 = m.median_method(test, 11)
m15 = m.median_method(test, 15)
m21 = m.median_method(test, 21)


decimated_vec = m.decimate_method(test, 100)
#divided_vec = m.divide_by_time(test, 0.05)


x1, y1 = m.change_to_vector(decimated_vec)
x2, y2 = m.change_to_vector(m21)


vectors = []
for i in range(len(x1)):
    vec = [x1[i], y1[i]]
    vectors.append(vec)
  #  print(vec)



#to do sprawdzenia czy sie zdecymowało
#print("Original data shape: ", test.shape)
#print("Decimated data shape: ", decimated_vec.shape)
#print(decimated_vec)


#to nie działa jak co bo są wartości nan w tym jest problem
#plt.plot(x1, y1, 'b.', label='L')
#plt.plot(x2, y2, 'r.', label='P')
#plt.title("Przebiegi parami")
#plt.xlabel("Czas")
#plt.ylabel("Wartości pomiarów")
#plt.legend()
#plt.show()













