# Mac 환경 데이터셋 다운로드 시 발생하는 SSL 인증서 오류 방지
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 텐서플로우 및 Keras 신경망, 이미지 처리, 수치 연산 라이브러리 로드
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. CIFAR-10 데이터셋 로드 및 훈련/테스트 세트 분할
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = tf.keras.datasets.cifar10.load_data()

# 2. 데이터 전처리: 픽셀 값을 0~1 범위로 정규화하여 모델의 학습 속도 향상
x_train_cifar, x_test_cifar = x_train_cifar / 255.0, x_test_cifar / 255.0

# 3. CNN 모델 설계
# Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 합성곱 신경망 구성 
cifar_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), # 32개 필터, 3x3 커널, 입력 32x32 컬러 이미지
    MaxPooling2D((2, 2)),                                           # 2x2 영역에서 가장 큰 값만 남겨 이미지 크기 축소
    Conv2D(64, (3, 3), activation='relu'),                          # 64개 필터 적용하여 특징 추출
    MaxPooling2D((2, 2)),                                           # 다시 한 번 크기 축소
    Conv2D(64, (3, 3), activation='relu'),                          # 64개 필터 적용
    Flatten(),                                                      # 2차원 특징 맵을 1차원 배열로 평탄화
    Dense(64, activation='relu'),                                   # 64개 노드의 은닉층 통과
    Dense(10, activation='softmax')                                 # 10개의 클래스에 대한 확률을 출력하는 출력층
])

# 모델 컴파일: 최적화 알고리즘, 손실 함수, 평가 지표 설정
cifar_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# 모델 훈련: 10번(epochs=10) 반복 학습 수행
print("--- CIFAR-10 CNN 모델 훈련 시작 ---")
cifar_model.fit(x_train_cifar, y_train_cifar, epochs=10, validation_data=(x_test_cifar, y_test_cifar))

# 4. 모델 성능 평가
print("\n--- CIFAR-10 모델 평가 ---")
cifar_loss, cifar_acc = cifar_model.evaluate(x_test_cifar, y_test_cifar, verbose=2)

# 최종 테스트 정확도 출력
print(f"CIFAR-10 테스트 정확도: {cifar_acc:.4f}\n")

# 5. 테스트 이미지(dog.jpg) 예측 함수 정의
def predict_custom_image(img_path='dog.jpg'):
    try:
        # 모델 입력에 맞춰 이미지를 32x32 크기로 로드
        img = image.load_img(img_path, target_size=(32, 32))
        
        # 불러온 이미지를 배열 형태로 변환
        img_array = image.img_to_array(img)
        
        # 모델 입력 형식에 맞추기 위해 배치 차원(1차원) 추가
        img_array = np.expand_dims(img_array, axis=0) 
        
        # 학습 데이터와 동일하게 픽셀 값을 0~1로 정규화
        img_array /= 255.0 
        
        # 이미지를 모델에 넣어 클래스별 확률 예측
        predictions = cifar_model.predict(img_array)
        
        # 가장 확률이 높은 클래스의 인덱스 추출
        predicted_class_index = np.argmax(predictions[0])
        
        # CIFAR-10 데이터셋의 10가지 클래스 이름 목록
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 최종 예측 결과 출력 [cite: 36]
        print(f"테스트 이미지 '{img_path}'의 예측 결과: {class_names[predicted_class_index]}")
        
    except FileNotFoundError:
        # 이미지가 같은 폴더에 없을 경우의 예외 처리
        print(f"오류: '{img_path}' 파일을 찾을 수 없습니다. 코드를 실행하는 위치에 파일이 존재하는지 확인해주세요.")

# dog.jpg 이미지에 대한 예측 실행
print("--- dog.jpg 이미지 예측 ---")
predict_custom_image('dog.jpg')