import cv2
import dlib
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

# Dlib의 눈 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 상태를 분류하기 위한 사전 훈련된 모델 로드 (예: TensorFlow/Keras 모델)
model = load_model('eye_state_model.h5')

def detect_closed_eyes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        
        # 눈 좌표 추출
        left_eye_x = left_eye.x
        left_eye_y = left_eye.y
        right_eye_x = right_eye.x
        right_eye_y = right_eye.y
        
        # 이미지에서 눈 영역 추출
        left_eye_image = gray[left_eye_y-10:left_eye_y+10, left_eye_x-10:left_eye_x+10]
        right_eye_image = gray[right_eye_y-10:right_eye_y+10, right_eye_x-10:right_eye_x+10]

        if is_eye_closed(left_eye_image) or is_eye_closed(right_eye_image):
            return True
    
    return False

def is_eye_closed(eye_image):
    eye_image = cv2.resize(eye_image, (24, 24))  # 모델 입력 크기에 맞게 조정
    eye_image = eye_image / 255.0  # 정규화
    eye_image = np.expand_dims(eye_image, axis=-1)  # 모델에 따라 필요할 수 있음
    
    eye_image = np.expand_dims(eye_image, axis=0)
    prediction = model.predict(eye_image)
    return np.argmax(prediction) == 1  # 1은 눈 감은 상태, 0은 눈 뜬 상태로 가정

# 이미지 경로 목록 설정 (디렉토리 내 모든 이미지 파일 경로 추가)
image_dir = r'C:\Users\kdhhi\Pictures\dataset1'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# 눈 감은 사진 선택
closed_eye_images = [image_path for image_path in image_paths if detect_closed_eyes(image_path)]
print('눈 감은 사진:', closed_eye_images)
