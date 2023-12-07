import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

# načtení CIFAR-10 datasetu
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# předpřipravení dat
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# postavení modelu
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# compilace dat
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# trénování modelu
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluace dat
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')