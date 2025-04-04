import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import pandas as pd
from keras import layers, models
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt

BASE_PATH = "./ISIC 2024/isic-2024-challenge"
image_dir = os.path.join(BASE_PATH, "train-image", "image")
try:
    df = pd.read_csv(os.path.join(BASE_PATH, 'train-metadata.csv'), 
                   low_memory=False,
                   encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(os.path.join(BASE_PATH, 'train-metadata.csv'), 
                   low_memory=False,
                   encoding='latin1')
df = df.ffill()

# extrair imagens e rótulos
images = []
labels = []

for index, row in df.head(25000).iterrows():
    print(index)
    try:
        img_path = os.path.join(image_dir, row['isic_id'] + '.jpg')  # ou .png
        img = cv2.imread(img_path)
        if img is None:
            print(f"Aviso: não foi possivel ler a imagem {img_path}")
            continue
        
        img = cv2.resize(img, (128, 128))  # redimensionar as imagens
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mudar padrão de bgr para rgb
        images.append(img)
        labels.append(row['target'])
    except Exception as e:
        print(f"Erro ao processar imagem {index}: {str(e)}")
        continue

# converter para arrays numpy
X = np.array(images)
y = np.array(labels)

print(f"Final data shape - X: {X.shape}, y: {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify = y)

model = models.Sequential([
    Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # usado para classificação binária
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# 1. Treinamento do Modelo
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

val_loss, val_accuracy, val_auc  = model.evaluate(X_val, y_val)
print(f'Loss: {val_loss}, Accuracy: {val_accuracy}, AUC: {val_auc}')

y_pred = model.predict(X_val)

# calculo da AUC manualmente com roc_auc_score
auc_manual = roc_auc_score(y_val, y_pred)
print(f'Manual AUC: {auc_manual}')

# histórico pros gráficos
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_auc = history.history['auc']
val_auc = history.history['val_auc']

# gráficos de Acurácia, Loss e AUC
plt.figure(figsize=(18, 6))

# Gráfico de Acurácia
plt.subplot(1, 3, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfico de Loss
plt.subplot(1, 3, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Gráfico de AUC
plt.subplot(1, 3, 3)
plt.plot(train_auc, label='Train AUC')
plt.plot(val_auc, label='Validation AUC')
plt.title('AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

plt.show()
