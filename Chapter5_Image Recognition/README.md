## 🌀 문제 01 간단한 이미지 분류기 구현


<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/997d377a-dee3-4ce2-9f3f-04f8233bfe58" />



### 1.주요 개념

- **정규화 (Normalization)**: 이미지의 픽셀 값은 원래 0(검은색)부터 255(흰색) 사이의 숫자로 되어 있습니다. 이 큰 숫자들을 0~1 사이의 작은 숫자로 줄여주는 작업.
- **다층 퍼셉트론 (MLP)**: Sequential 모델과 Dense 레이어를 활용하여 구성한 가장 기본적인 형태의 인공신경망.
- **평탄화 (Flatten)**: 우리가 만든 기본 신경망(Dense 레이어)은 1차원 형태의 한 줄짜리 데이터만 받아들일 수 있기에 28x28 형태의 2차원 네모난 이미지를 784(28x28)개의 숫자가 일렬로 늘어선 1차원 배열로 쭉 펴주는 작업.
- **활성화 함수 (Activation Function)**: 뉴런이 다음 뉴런으로 신호를 보낼지 말지 결정하는 함수.
  -   ReLU: 음수 값은 0으로 버리고, 양수 값만 그대로 통과시켜 학습 효율을 극대화함.
  -   Softmax: 모델이 내놓은 최종 결괏값들을 모두 합쳐서 100%(1.0)가 되도록 만들어 줌.



### 2. 전체 코드

```Python
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
```


### 3. 핵심 코드
- **`(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()`** : load_data() 함수를 호출하면 방대한 손글씨 데이터를 인공지능이 공부할 (train)과 (test)로 나누어 줌.
  
- **`x_train_mnist = x_train_mnist / 255.0`** : 가장 큰 픽셀 값인 255로 전체 데이터를 나누어, 모든 값을 0에서 1 사이로 압축하는 과정.
  
- **`mnist_model = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') 
])`** :
  - Sequential: 레이어들을 순서대로 차곡차곡 쌓음.
  - Flatten: 28x28 이미지를 한 줄로 평탄화함.
  - Dense(128): 128개의 뉴런을 가진 학습용 뇌(은닉층)를 만듦.
  - Dense(10): 최종적으로 0~9까지 10개의 숫자 중 하나를 골라야 하므로 출력층의 뉴런을 10개로 설정.

- **'mnist_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')'**'
  - optimizer: 모델이 정답을 찾아가는 방향을 조절하는 내비게이션(최적화 알고리즘)입니다. 'adam'은 가장 무난하고 성능이 좋은 방식.
  - loss: 모델의 예측이 정답과 얼마나 틀렸는지 채점하는 기준(손실 함수)입니다. 다중 분류 문제에서는 주로 'categorical_crossentropy'를 사용.




### 4. 실행 결과


<img width="573" height="216" alt="result1" src="https://github.com/user-attachments/assets/21bd5841-e752-4b50-a5dd-28e6ed2ef64f" />


---

## 🌀 문제 02 CIFAR-10 데이터셋을 활용한 CNN 모델 구축



<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/7a9fbbcb-da65-4f5c-88b3-576cfef1cbd6" />



### 1. 주요 개념

- **CNN (합성곱 신경망 - Convolutional Neural Network)**: 일반 신경망(Dense)은 이미지를 1차원(한 줄)으로 길게 펴서 학습하기 때문에 '모양'이나 '위치' 같은 공간적인 특징을 잃어버립니다. 반면 CNN은 2차원 이미지 형태를 그대로 유지하면서 특징을 뽑아낸다.
- **합성곱 층 (Convolutional Layer / Conv2D)**: 이미지 위를 작은 돋보기(필터/커널)가 훑고 지나가면서 선, 질감, 색상 같은 특징(Feature)을 도장 찍듯 추출하는 역할을 함.
- **풀링 층 (Pooling Layer / MaxPooling2D)**: 합성곱 층에서 찾은 특징들 중에서 가장 강하고 중요한 특징만 남기고 크기를 반으로 줄이는(요약하는) 과정으로 데이터 크기를 줄여 계산량을 감소시키고, 이미지가 약간 찌그러지거나 이동해도 잘 인식할 수 있게 해준다.


### 2. 전체 코드

```Python
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
```


### 3. 핵심 코드

- **`cifar_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    ...
])`** :
  - Conv2D(32, (3, 3)): 3x3 크기의 돋보기(필터) 32개를 사용해서 이미지의 특징을 찾아낸다. input_shape=(32, 32, 3)에서 끝의 '3'은 컬러 이미지의 R, G, B 세 가지 색상 채널을 의미한다.
  - MaxPooling2D((2, 2)): 2x2 크기의 영역에서 가장 큰(가장 특징적인) 값 하나만 뽑아내어 이미지를 요약한다.

- **` Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')`** :
  - 앞선 Conv2D와 MaxPooling2D를 거치며 추출된 '특징 맵(Feature Map)'을 Flatten()으로 평탄화 함.
  - 일반적인 신경망인 Dense 레이어에 통과시켜 이 특징들이 10개의 클래스 중 어디에 가장 가까운지 최종 확률을 계산함.
  
- **`img = image.load_img('dog.jpg', target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array /= 255.0`** :
  - load_img(..., target_size=(32, 32)): 우리가 가진 강아지 사진의 크기가 제각각일 테니, 모델이 공부했던 크기인 32x32로 강제로 맞춰서 불러옴.
  - img_to_array: 불러온 이미지를 컴퓨터가 계산할 수 있는 숫자 배열로 바꿔줌.
  - np.expand_dims: 모델은 항상 '여러 장의 이미지 묶음(배치)'을 받도록 설계되어 있습니다. 1장만 넣더라도 [1장짜리 묶음] 형태로 차원을 하나 늘려줌.
  - /= 255.0: 모델이 공부할 때 0~1 사이 값으로 정규화해서 배웠으므로, 시험 문제(새로운 사진)도 똑같은 스케일로 맞춰줌.



### 4. 실행 결과


<img width="860" height="409" alt="image" src="https://github.com/user-attachments/assets/86d55ed9-a8c9-45e5-9dc8-2d395da8daeb" />


---

