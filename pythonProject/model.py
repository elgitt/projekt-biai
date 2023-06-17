import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import methods as m


def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')  # 6 klas
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def prepare_data():
    data = m.get_vectors()

    vec0_data = np.array(data['vec0'])
    vec1_data = np.array(data['vec1'])
    vec2_data = np.array(data['vec2'])
    vec3_data = np.array(data['vec3'])
    vec4_data = np.array(data['vec4'])
    vec5_data = np.array(data['vec5'])

    x_train = np.concatenate((vec0_data, vec1_data, vec2_data, vec3_data, vec4_data, vec5_data), axis=0)
    y_train = np.concatenate((
        np.tile([1, 0, 0, 0, 0, 0], (vec0_data.shape[0], 1)),  # pwm150
        np.tile([0, 1, 0, 0, 0, 0], (vec1_data.shape[0], 1)),  # pwm100
        np.tile([0, 0, 1, 0, 0, 0], (vec2_data.shape[0], 1)),  # pwm100 z obciążeniem
        np.tile([0, 0, 0, 1, 0, 0], (vec3_data.shape[0], 1)),  # pwm50
        np.tile([0, 0, 0, 0, 1, 0], (vec4_data.shape[0], 1)),  # pwm200
        np.tile([0, 0, 0, 0, 0, 1], (vec5_data.shape[0], 1)),  # pwm200 z obciążeniem
    ), axis=0)

    return x_train, y_train


def train_model():
    x_train, y_train = prepare_data()
    input_shape = x_train[0].shape

    model = create_model(input_shape)
    model.fit(x_train, y_train, epochs=5000, batch_size=100)

    model.save("test3.h5")

    loaded_model = load_model("test3.h5")
    _, accuracy = loaded_model.evaluate(x_train, y_train)

    if accuracy >= 0.95:
        print("Model has been successfully trained!")
    else:
        print("There was a problem during model training.")


def prepare_test_vector(vec):
    data = m.get_vectors()
    vec_test = np.array(data[vec])

    return vec_test


def predict_class(test_data):
    loaded_model = load_model("test3.h5")
    class_labels = {
        0: 'pwm150',
        1: 'pwm100',
        2: 'pwm100 z obciążeniem',
        3: 'pwm50',
        4: 'pwm200',
        5: 'pwm200 z obciążeniem',
    }

    predicted_class = loaded_model.predict(test_data)
    predicted_class_idx = np.argmax(predicted_class)
    if predicted_class_idx == 9:
        predicted_class_idx = 3
    id_type = int(predicted_class_idx)
    class_label = class_labels[id_type]
    print("class:", str(predicted_class_idx) + " " + class_label)

    df = pd.read_csv('filtered_data/Tek000' + str(predicted_class_idx) + '.csv')

    plt.plot(df['TIME'], df['CH1'], label='Current')
    plt.plot(df['TIME'], df['CH2'], label='Voltage')
    plt.xlabel('Time')
    plt.ylabel('Current and Voltage')
    plt.title('Plot of Current and Voltage over Time ' + class_label)
    plt.legend()
    plt.show()


