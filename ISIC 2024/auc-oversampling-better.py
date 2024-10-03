import numpy as np
import pandas as pd
import os
from keras import layers, models, regularizers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import RandomOverSampler

# Configurando o caminho base e carregando os dados
BASE_PATH = "./ISIC 2024/isic-2024-challenge"
image_dir = f"{BASE_PATH}/train-image/image"
df = pd.read_csv(f'{BASE_PATH}/train-metadata.csv')
df = df.ffill()

# Extraindo imagens e rótulos
images = []
labels = []

for index, row in df.head(25000).iterrows():
    print(index)
    img_path = os.path.join(image_dir, row['isic_id'] + '.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))  # Redimensionando as imagens
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Mudando padrão de BGR para RGB
    images.append(img)
    labels.append(row['target'])

# Convertendo para arrays numpy
X = np.array(images)
y = np.array(labels)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Realizando oversampling antes do train-test split
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = ros.fit_resample(X.reshape(X.shape[0], -1), y)  # Reajustando para 2D

# Convertendo de volta para o formato original
X_resampled = X_resampled.reshape(-1, 64, 64, 3)

# Dividindo os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled)

class_weights = {0: (1 / np.sum(y_train == 0)) * (len(y_train) / 2.0),
                 1: (1 / np.sum(y_train == 1)) * (len(y_train) / 2.0)}

# Contagem e porcentagem no conjunto de treino
valores_train, contagens_train = np.unique(y_train, return_counts=True)
total_train = len(y_train)
print(f'Treino - Benigno (0): {contagens_train[0] if 0 in valores_train else 0} ({(contagens_train[0] / total_train) * 100:.2f}%), '
      f'Maligno (1): {contagens_train[1] if 1 in valores_train else 0} ({(contagens_train[1] / total_train) * 100:.2f}%)')

# Contagem e porcentagem no conjunto de validação
valores_val, contagens_val = np.unique(y_val, return_counts=True)
total_val = len(y_val)
print(f'Validação - Benigno (0): {contagens_val[0] if 0 in valores_val else 0} ({(contagens_val[0] / total_val) * 100:.2f}%), '
      f'Maligno (1): {contagens_val[1] if 1 in valores_val else 0} ({(contagens_val[1] / total_val) * 100:.2f}%)')

# Normalizando os dados de treino e validação
X_train = X_train / 255.0
X_val = X_val / 255.0

# Definindo o modelo com Dropout e regularização L2
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    
    # Aumentando a taxa de Dropout
    layers.Dropout(0.7),  # Experimente diferentes valores
    
    # Camada densa com regularização L2
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')  # Saída binária para classificação
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento com os dados normalizados e validando a cada época
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])

# Avaliação do modelo no conjunto de validação
val_loss, val_accuracy, val_auc = model.evaluate(X_val, y_val)
print(f'Loss: {val_loss}, Accuracy: {val_accuracy}, AUC: {val_auc}')

# Prevendo rótulos no conjunto de validação
y_pred = model.predict(X_val)

# Calculando a AUC manualmente com roc_auc_score (usando y_true e y_pred)
auc_manual = roc_auc_score(y_val, y_pred)
print(f'Manual AUC: {auc_manual}')

# Extraindo histórico para gráficos
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_auc = history.history['auc']
val_auc = history.history['val_auc']

# Plotando os gráficos de Acurácia, Loss e AUC
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
