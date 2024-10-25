import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

x = pickle.load(open("X.pk1", "rb"))
y = pickle.load(open("Y.pk1", "rb"))

x = x/255

model = Sequential()

model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, input_shape = (100, 100, 3)))

model.add(Dense(128, input_shape = (100, 100, 3)))

model.add(Dense(256, input_shape = (100, 100, 3)))

model.add(Dense(128, input_shape = (100, 100, 3)))

model.add(Dense(128, input_shape = (100, 100, 3)))

model.add(Dense(3, activation = "softmax"))

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x, y, epochs = 7, validation_split=0.001)

model.save("model.h5")