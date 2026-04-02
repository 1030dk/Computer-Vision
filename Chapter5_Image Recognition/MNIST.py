# Mac 환경의 데이터셋 다운로드 SSL 인증서 오류 방지
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 텐서플로우 및 인공신경망 구축에 필요한 케라스 모듈 로드
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 1. MNIST 데이터셋 로드 및 훈련/테스트 세트 분할
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

# 2. 데이터 전처리: 0~255 픽셀 값을 0~1 사이로 정규화 (28x28 흑백 이미지)
x_train_mnist, x_test_mnist = x_train_mnist / 255.0, x_test_mnist / 255.0

# 3. 간단한 신경망 모델 구축
# Sequential 모델과 Dense 레이어를 활용
mnist_model = Sequential([
    Flatten(input_shape=(28, 28)),    # 28x28 2차원 이미지를 1차원 배열로 평탄화
    Dense(128, activation='relu'),    # 128개의 노드를 가진 은닉층 (활성화 함수: ReLU)
    Dense(10, activation='softmax')   # 0~9의 숫자 10개를 분류하는 출력층 (활성화 함수: Softmax)
])

# 4. 모델 컴파일: 최적화 알고리즘, 손실 함수, 평가 지표 설정
mnist_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# 5. 모델 훈련 (epochs 5회 반복) 
print("--- MNIST 모델 훈련 시작 ---")
mnist_model.fit(x_train_mnist, y_train_mnist, epochs=5)

# 6. 테스트 세트를 이용한 모델 정확도 평가
print("\n--- MNIST 모델 평가 ---")
mnist_loss, mnist_acc = mnist_model.evaluate(x_test_mnist, y_test_mnist, verbose=2)

# 최종 정확도 결과 출력
print(f"MNIST 테스트 정확도: {mnist_acc:.4f}\n")