import numpy as np
import tensorflow as tf
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

    # Przygotowanie danych treningowych
    X_train = np.concatenate((vec0_data, vec1_data, vec2_data, vec3_data, vec4_data, vec5_data), axis=0)
    y_train = np.concatenate((
        np.tile([1, 0, 0, 0, 0, 0], (vec0_data.shape[0], 1)),  # pwm150
        np.tile([0, 1, 0, 0, 0, 0], (vec1_data.shape[0], 1)),  # pwm100
        np.tile([0, 0, 1, 0, 0, 0], (vec2_data.shape[0], 1)),  # pwm100 z obciążeniem
        np.tile([0, 0, 0, 1, 0, 0], (vec3_data.shape[0], 1)),  # pwm50
        np.tile([0, 0, 0, 0, 1, 0], (vec4_data.shape[0], 1)),  # pwm200
        np.tile([0, 0, 0, 0, 0, 1], (vec5_data.shape[0], 1)),  # pwm200 z obciążeniem
    ), axis=0)

    return X_train, y_train


def prepare_test_vector():
    data = m.get_vectors()

    vec_test = np.array(data['vec4'])

    test_vector = {
        'vec_test': vec_test,
    }

    return test_vector



