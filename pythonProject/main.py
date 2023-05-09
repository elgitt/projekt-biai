import methods as m
import matplotlib.pyplot as plt
import numpy as np

test = m.read_from_parquet('0kg_L')

# m.return_median_mx('0kg_L')['parquet']
#test2 = m.return_median_mx('0kg_R')['parquet']

decimated_vec = m.decimate_method(test, 100)
#decimated_vec2 = m.decimate_method(test2, 100)

#x1, y1 = m.change_to_vector(decimated_vec)
#x2, y2 = m.change_to_vector(decimated_vec2)

test = decimated_vec.head(-1)
#test2 = decimated_vec2.head(-1)

plt.scatter(test["Czas"], test["CH1"], label='L')
#plt.scatter(test2["Czas"], test2["CH1"], label='P')
#plt.plot(x2, y2, 'r.', label='P')
plt.title("Przebiegi parami")
plt.xlabel("Czas")
plt.ylabel("Wartości pomiarów")
#plt.ylim(0, 1)
#plt.xlim(-2, 6)
plt.yticks(np.arange(0, 0.1, 100))
plt.legend()
plt.show()

# wektor z wektorow
# vectors = []
# for i in range(len(x1)):
#   vec = [x1[i], y1[i]]
#  vectors.append(vec)
#  print(vec)


# print("Original data shape: ", test.shape)
# print("Decimated data shape: ", decimated_vec.shape)
# print(decimated_vec)
