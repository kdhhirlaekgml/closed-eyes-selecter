import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import logging

# 경고 메시지 무시 설정
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_images_from_folder(folder, label, image_size=(24, 24)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            # 파일 경로 인코딩 문제 해결
            img_path_encoded = img_path.encode('utf-8').decode('utf-8')
            img = cv2.imread(img_path_encoded, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
            else:
                print(f"Failed to load image: {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

# 경로 설정
open_eyes_folder = r"C:\Users\kdhhi\Pictures\open_eyes"
closed_eyes_folder = r"C:\Users\kdhhi\Pictures\closed_eyes"

open_eyes_images, open_eyes_labels = load_images_from_folder(open_eyes_folder, 0)
closed_eyes_images, closed_eyes_labels = load_images_from_folder(closed_eyes_folder, 1)

# 이미지와 라벨 합치기
images = np.array(open_eyes_images + closed_eyes_images)
labels = np.array(open_eyes_labels + closed_eyes_labels)

# 데이터 정규화
images = images / 255.0
images = np.expand_dims(images, axis=-1)

# 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 모델 학습
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=10)

# 모델 저장
model.save('eye_state_model.h5')
