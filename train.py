import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load your data
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
    fill_mode='nearest')

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=x.shape[1:])


x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(3, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=10,
          validation_data=(x_val, y_val))


test_loss, test_acc = model.evaluate(x_val, y_val)
print('Test accuracy:', test_acc)

model.save('model_resnet.h5')
