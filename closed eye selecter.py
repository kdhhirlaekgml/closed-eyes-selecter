import cv2
import dlib
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf

# TensorFlow 로그 레벨을 설정하여 경고 메시지를 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU 장치 비활성화 (GPU가 없거나 설정되지 않은 경우)
tf.config.set_visible_devices([], 'GPU')

# Dlib의 얼굴 검출기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 상태 분류를 위한 사전 훈련된 모델 로드
model = load_model('eye_state_model.h5')

def detect_closed_eyes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 읽기 오류: {image_path}")
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        left_eye_image = extract_eye_image(gray, left_eye_coords)
        right_eye_image = extract_eye_image(gray, right_eye_coords)
        
        if is_eye_closed(left_eye_image) or is_eye_closed(right_eye_image):
            return True
    
    return False

def extract_eye_image(gray, eye_coords):
    x_coords = [x for x, y in eye_coords]
    y_coords = [y for x, y in eye_coords]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    eye_image = gray[min_y:max_y, min_x:max_x]
    return eye_image

def is_eye_closed(eye_image):
    eye_image = cv2.resize(eye_image, (24, 24))  # 모델 입력 크기에 맞게 조정
    eye_image = eye_image / 255.0  # 정규화
    eye_image = np.expand_dims(eye_image, axis=-1)  # 필요시 채널 차원 추가
    eye_image = np.expand_dims(eye_image, axis=0)  # 배치 차원 추가
    
    prediction = model.predict(eye_image)
    return np.argmax(prediction) == 1  # 1은 눈 감은 상태, 0은 눈 뜬 상태로 가정

# 이미지가 포함된 디렉토리 설정
image_dir = r'C:\Users\kdhhi\Pictures\dataset1'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 눈 감은 사진 선택
closed_eye_images = [image_path for image_path in image_paths if detect_closed_eyes(image_path)]
print('눈 감은 사진:', closed_eye_images)
