import numpy as np
import pandas as pd
import os
from keras import  layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
os.environ["KERAS_BACKEND"] = "tensorflow"
import cv2
import matplotlib.pyplot as plt 

BASE_PATH = "./ISIC 2024/isic-2024-challenge"
image_dir = f"{BASE_PATH}/train-image/image"
df = pd.read_csv(f'{BASE_PATH}/train-metadata.csv')
df = df.ffill()

# extrair imagens e rótulos
images = []
labels = []

for index, row in df.head(25000).iterrows():
    print(index)
    img_path = os.path.join(image_dir, row['isic_id'] + '.jpg')  # ou .png
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # redimensionar as imagens
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mudar padrão de bgr para rgb
    images.append(img)
    labels.append(row['target']) 

# converter para arrays numpy
X = np.array(images)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify = y)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # usado para classificação binária
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 1. Treinamento do Modelo
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Loss: {val_loss}, Accuracy: {val_accuracy}')

# 2. Verificar Overfitting e 3. Visualizar o Desempenho
# Acessar as métricas de precisão e perda do histórico
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))

# Gráfico de precisão
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()