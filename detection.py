
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from skimage import exposure, color
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# načtení CIFAR-10 datasetu
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Use a smaller subset for testing
subset_size = 1000
train_images, train_labels = train_images[:subset_size], train_labels[:subset_size]
test_images, test_labels = test_images[:subset_size], test_labels[:subset_size]


# Load CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to display images with their predicted labels
def display_images(images, labels_true, labels_pred, class_names, num_images=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(f'True: {class_names[labels_true[i]]}\nPred: {class_names[labels_pred[i]]}')
        plt.axis('off')
    plt.show()

# Function to save images with their predicted labels
def save_images(images, labels_true, labels_pred, class_names, output_dir='output_images'):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        plt.imshow(img)
        plt.title(f'True: {class_names[labels_true[i]]}\nPred: {class_names[labels_pred[i]]}')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'image_{i + 1}.png'))
        plt.close()


def extract_single_hog(image):
    gray_image = color.rgb2gray(image)
    fd, _ = hog(gray_image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
    return fd

def extract_hog_features_parallel(images):
    hog_features = Parallel(n_jobs=-1)(delayed(extract_single_hog)(image) for image in images)
    return hog_features

# Extraktování HOG funkcí pro trénování a testování
start_time_extraction = time.time()
train_images_hog_parallel = extract_hog_features_parallel(train_images)
test_images_hog_parallel = extract_hog_features_parallel(test_images)
end_time_extraction = time.time()

# Flatten HOG funkce
train_images_hog_flat = [img.flatten() for img in train_images_hog_parallel]
test_images_hog_flat = [img.flatten() for img in test_images_hog_parallel]

# Flatten popisky pro SVM
train_labels_flat = train_labels.flatten()
test_labels_flat = test_labels.flatten()

# Trénování HOG + SVM modelu
svm_model = SVC()
start_time_training = time.time()
svm_model.fit(train_images_hog_flat, train_labels_flat)
end_time_training = time.time()

# Predikování s HOG + SVM modelem
start_time_prediction = time.time()
predictions_hog = svm_model.predict(test_images_hog_flat)
end_time_prediction = time.time()

# Zobrazení a uložení HOG + SVM výsledků
start_time_displayH = time.time()
display_images(test_images, test_labels_flat, predictions_hog, class_names)
save_images(test_images, test_labels_flat, predictions_hog, class_names, output_dir='hog_svm_output_images')
end_time_displayH = time.time()

# Evaluace HOG + SVM modelu
accuracy_hog = accuracy_score(test_labels_flat, predictions_hog)
print(f'HOG + SVM Test accuracy: {accuracy_hog}')

print(f'Time taken for HOG feature extraction: {end_time_extraction - start_time_extraction:.2f} seconds')
print(f'Time taken for SVM model training: {end_time_training - start_time_training:.2f} seconds')
print(f'Time taken for HOG + SVM prediction: {end_time_prediction - start_time_prediction:.2f} seconds')
print(f'Time taken for HOG + SVM to display: {end_time_displayH - start_time_displayH:.2f} seconds')

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
start_time_cnn_training = time.time()
model_cnn.fit(train_images_cnn, train_labels_cnn, epochs=10, validation_data=(test_images_cnn, test_labels_cnn), verbose=2)
end_time_cnn_training = time.time()


# predikování pomocí CNN modelu
start_time_cnn_prediction = time.time()
predictions_cnn_probs = model_cnn.predict(test_images_cnn)
predictions_cnn = np.argmax(predictions_cnn_probs, axis=1)
end_time_cnn_prediction = time.time()

# Zobrazení a uložení CNN výsledků
start_time_displayC = time.time()
display_images(test_images, test_labels_flat, predictions_cnn, class_names)
save_images(test_images, test_labels_flat, predictions_cnn, class_names, output_dir='cnn_output_images')
end_time_displayC = time.time()

# Evaluace CNN modelu
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(test_images_cnn, test_labels_cnn)
print(f'CNN Test accuracy: {test_acc_cnn}')

print(f'Time taken for CNN model training: {end_time_cnn_training - start_time_cnn_training:.2f} seconds')
print(f'Time taken for CNN prediction: {end_time_cnn_prediction - start_time_cnn_prediction:.2f} seconds')
print(f'Time taken for CNN display: {end_time_displayC - start_time_displayC:.2f} seconds')


