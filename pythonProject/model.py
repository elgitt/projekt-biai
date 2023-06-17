from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape)) # Warstwa wejściowa
    model.add(Dense(32, activation='relu'))  # Dodatkowe warstwy (opcjonalnie)
    model.add(Dense(6, activation='softmax'))  # Warstwa wyjściowa z sześcioma neuronami

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

