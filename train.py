import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split






x = pickle.load(open("X.pk1", "rb"))
y = pickle.load(open("Y.pk1", "rb"))


x = x.astype('float32')  
x /= 255.0  

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


datagen = ImageDataGenerator(
    rotation_range=40,  
    width_shift_range=0.2, 
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' 
)


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x.shape[1:]))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())


model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax')) 


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=10)

test_loss, test_acc = model.evaluate(x_val, y_val)
print('Test accuracy:', test_acc)

model.save('model.h5')
