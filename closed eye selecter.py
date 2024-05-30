import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 얼굴과 눈을 검출하기 위해 OpenCV의 Haar Cascade 사용
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 눈 상태를 분류하기 위한 사전 훈련된 모델 로드 (예: TensorFlow/Keras 모델)
model = load_model('eye_state_model.h5')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화로 대비 향상
    return gray

def detect_faces_and_eyes(image):
    gray = preprocess_image(image)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    eye_images = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        
        for (ex, ey, ew, eh) in eyes:
            eye_image = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_image = cv2.resize(eye_image, (24, 24))  # 모델 입력 크기에 맞게 조정
            eye_image = eye_image / 255.0  # 정규화
            eye_image = np.expand_dims(eye_image, axis=-1)  # 모델에 따라 필요할 수 있음
            eye_images.append(eye_image)

    return eye_images

def is_eye_closed(eye_image):
    eye_image = np.expand_dims(eye_image, axis=0)
    prediction = model.predict(eye_image)
    return np.argmax(prediction) == 1  # 1은 눈 감은 상태, 0은 눈 뜬 상태로 가정

def process_images(image_paths):
    closed_eye_images = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image {image_path}")
            continue
        
        eye_images = detect_faces_and_eyes(image)
        if not eye_images:
            print(f"No eyes detected in {image_path}")
        
        for eye_image in eye_images:
            if is_eye_closed(eye_image):
                closed_eye_images.append(image_path)
                break

    return closed_eye_images

# 이미지 경로 목록 설정 (디렉토리 내 모든 이미지 파일 경로 추가)
image_dir = r'C:\Users\kdhhi\Pictures\dataset1'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# 눈 감은 사진 선택
closed_eye_images = process_images(image_paths)
print('눈 감은 사진:', closed_eye_images)
