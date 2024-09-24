import os
os.environ["KERAS_BACKEND"] = "tensorflow" # outras opções: tensorflow ou torch

from keras import ops
import tensorflow as tf
import matplotlib.pyplot as plt 

import cv2
import pandas as pd
import numpy as np

BASE_PATH = "./ISIC 2024/isic-2024-challenge"

from IPython.display import display

# treino + validação
df = pd.read_csv(f'{BASE_PATH}/train-metadata.csv')
df = df.ffill()
display(df.head(2))

# teste
testing_df = pd.read_csv(f'{BASE_PATH}/test-metadata.csv')
testing_df = testing_df.ffill()
display(testing_df.head(2))




# from sklearn.utils.class_weight import compute_class_weight


# labels = df['benign_malignant'].values

# # Converter labels para números (0: benign, 1: malignant)
# label_map = {'benign': 0, 'malignant': 1}
# labels = np.array([label_map[label] for label in labels])

# # como o dataset tem muitos negativos é feito o balanceamento para equilibrar a quantidade dos exemplos
# class_weights = compute_class_weight('balanced', classes=np.unique(df['target']), y=df['target'])
# class_weights = dict(enumerate(class_weights))
# print("Class Weights:", class_weights)



# carregando string de bytes das imagens
import h5py

training_validation_hdf5 = h5py.File(f"{BASE_PATH}/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"{BASE_PATH}/test-image.hdf5", 'r')

# vizualizando uma imagem de teste

isic_id = df.isic_id.iloc[0]

# Image as Byte String
byte_string = training_validation_hdf5[isic_id][()]
print(f"Byte String: {byte_string[:20]}....")

# Convert byte string to numpy array
nparr = np.frombuffer(byte_string, np.uint8)

# print("Image:")
# image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # reverse last axis for bgr -> rgb
# plt.imshow(image)



image_dir = f"{BASE_PATH}/train-image/image"

# extrair imagens e rótulos
images = []
labels = []

for index, row in df.head(10).iterrows():
    img_path = os.path.join(image_dir, row['isic_id'] + '.jpg')  # ou .png
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # redimensionar as imagens
    images.append(img)
    labels.append(row['target']) 

# Converter para arrays numpy
X = np.array(images)
y = np.array(labels)
print(X[1])
print(y[1])