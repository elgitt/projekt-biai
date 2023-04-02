import methods as m
import matplotlib.pyplot as plt

test = m.return_median_mx('test')['parquet']

decimated_vec = m.decimate_method(test, 100)
decimated_vec2 = m.decimate_method(test, 50)

x1, y1 = m.change_to_vector(decimated_vec)
x2, y2 = m.change_to_vector(decimated_vec2)

# nie pokazuje jednego
plt.plot(x1, y1, 'b.', label='L')
plt.plot(x2, y2, 'r.', label='P')
plt.title("Przebiegi parami")
plt.xlabel("Czas")
plt.ylabel("Wartości pomiarów")
plt.ylim(-2, 6)
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
