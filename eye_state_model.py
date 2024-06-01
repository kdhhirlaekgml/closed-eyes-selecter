import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import imgaug.augmenters as iaa

# 경고 메시지 무시 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 데이터 증강을 위한 이미지 변환기 초기화
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # 좌우 반전
    iaa.Affine(rotate=(-10, 10))  # 이미지 회전
])

def load_images_from_folder(folder, label, image_size=(24, 24)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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

# 데이터 증강 및 이미지 저장
augmented_images = []
augmented_labels = []

for img, label in zip(images, labels):
    augmented_img = augmenter.augment_image(img)
    augmented_images.append(augmented_img)
    augmented_labels.append(label)
    # 추가된 부분: 증강된 이미지를 저장합니다.
    cv2.imwrite(f"augmented_images/{len(augmented_images)}.jpg", augmented_img)

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(augmented_images, augmented_labels, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 출력 뉴런 수 변경
])

# 모델 컴파일
model.compile(optimizer='adam', 
              loss='binary_crossentropy',  # 손실 함수 변경
              metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# 모델 저장
model.save('eye_state_model.h5')
