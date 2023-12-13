import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from skimage import exposure
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# načtení CIFAR-10 datasetu
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# předpřipravování dat pro CNN
train_images_cnn, test_images_cnn = train_images / 255.0, test_images / 255.0
train_labels_cnn = to_categorical(train_labels, 10)
test_labels_cnn = to_categorical(test_labels, 10)

# budování CNN modelu
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# kompilování CNN modelu
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trénování CNN modelu
model_cnn.fit(train_images_cnn, train_labels_cnn, epochs=10, validation_data=(test_images_cnn, test_labels_cnn), verbose=2)

# Evaluace CNN modelu
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(test_images_cnn, test_labels_cnn)
print(f'CNN Test accuracy: {test_acc_cnn}')

# Předpřipravení dat pro HOG + SVM
def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray_image = tf.image.rgb_to_grayscale(image).numpy()[:, :, 0]
        fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_features.append(fd)
    return hog_features

# Extraktování HOG funkcí pro trénování a testování
train_images_hog = extract_hog_features(train_images)
test_images_hog = extract_hog_features(test_images)

# Flatten HOG funkce
train_images_hog_flat = [img.flatten() for img in train_images_hog]
test_images_hog_flat = [img.flatten() for img in test_images_hog]

# Flatten labels pro SVM
train_labels_flat = train_labels.flatten()
test_labels_flat = test_labels.flatten()

# Trénování HOG + SVM modelu
svm_model = SVC()
svm_model.fit(train_images_hog_flat, train_labels_flat)

# Predikování s HOG + SVM modelem
predictions_hog = svm_model.predict(test_images_hog_flat)

# Evaluace HOG + SVM modelu
accuracy_hog = accuracy_score(test_labels_flat, predictions_hog)
print(f'HOG + SVM Test accuracy: {accuracy_hog}')
