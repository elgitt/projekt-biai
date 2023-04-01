import methods as m


test = m.read_from_parquet('test')
m5 = m.median_method(test, 5)
m9 = m.median_method(test, 9)
m11 = m.median_method(test, 11)
m15 = m.median_method(test, 15)
m21 = m.median_method(test, 21)

x1, y1 = m.change_to_vector(m5)
vectors = []
for i in range(len(x1)):
    vec = [x1[i], y1[i]]
    vectors.append(vec)
    print(vec)













