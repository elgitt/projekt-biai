import methods as m


#m.convert_csv_to_bin('0kg_L', 'test')

test = m.refactor_binary_to_data_frame('test')
x1, y1 = m.change_to_vector(test)
vectors = []
for i in range(len(x1)):
    vec = [x1[i], y1[i]]
    vectors.append(vec)
    print(vec)













