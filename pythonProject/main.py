import numpy as np
import tensorflow as tf
import model as mod
from tensorflow.keras.models import load_model

# data = m.get_vectors()
#
# vec0_data = np.array(data['vec0'])
# vec1_data = np.array(data['vec1'])
# vec2_data = np.array(data['vec2'])
# vec3_data = np.array(data['vec3'])
# vec4_data = np.array(data['vec4'])
# vec5_data = np.array(data['vec5'])

# print("vec0:" ,vec0_data)
# print("vec1:" , vec1_data)
# print("vec2:" , vec2_data)
# print("vec3:" , vec3_data)
# print("vec4:" , vec4_data)
# print("vec5:" , vec5_data)

# X_train, y_train = mod.prepare_data()
# input_shape = X_train[0].shape
#
# model = mod.create_model(input_shape)
# model.fit(X_train, y_train, epochs=500, batch_size=50)

# model.save("supermodel.h5")

loaded_model = load_model("supermodel.h5")

# accuracy = loaded_model.evaluate(X_train, y_train)[1]
# if 0.95 <= accuracy < 1.0:
#     print("Model został pomyślnie nauczony!")
# else:
#     print("Wystąpił problem podczas uczenia modelu.")

class_labels = {
    0: 'pwm150',
    1: 'pwm100',
    2: 'pwm100 z obciążeniem',
    3: 'pwm50',
    4: 'pwm200',
    5: 'pwm200 z obciążeniem',
}

test_vectors = mod.prepare_test_vector()

# Wybierz wektor testowy
test_data = tf.convert_to_tensor(test_vectors['vec_test'])

# Przewidywanie klasy dla wektora testowego
predicted_class = loaded_model.predict(test_data)
predicted_class_idx = np.argmax(predicted_class)

print("class: ", predicted_class_idx)





# old main -------------------------------------------------------------------------------------------------------------
# dataL = m.read_from_parquet('0kg_L')
# dataR = m.read_from_parquet('0kg_R')
# dataW = m.read_from_parquet('0kg_W')

# m.generate_graph(dataL, dataR, dataW)

# model.model_MLP(dataL, dataR, dataW)
# old main -------------------------------------------------------------------------------------------------------------
