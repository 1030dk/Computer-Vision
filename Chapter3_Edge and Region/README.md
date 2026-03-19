## 🌀 문제 01 소벨 에지 검출 및 결과 시각화



### 1.주요 개념



### 2. 전체 코드

```Python
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 


# 1. 이미지 불러오기 및 RGB/그레이스케일 변환
img1 = cv.imread('edgeDetectionImage.jpg') # 실습 이미지를 파일에서 읽어오기
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB) # Matplotlib 출력을 위해 BGR 색상을 RGB로 변환하기
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) # 에지 검출 연산을 위해 이미지를 그레이스케일(흑백)로 변환하기 


# 2. x축 및 y축 방향 소벨 에지 검출 (ksize는 3으로 설정)
sobelx = cv.Sobel(gray1, cv.CV_64F, 1, 0, ksize=3) # 소벨 필터로 x축(가로) 방향의 에지 검출하기 
sobely = cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=3) # 소벨 필터로 y축(세로) 방향의 에지 검출하기 


# 3. 에지 강도 계산 및 시각화를 위한 타입 변환
magnitude = cv.magnitude(sobelx, sobely) # x축과 y축 방향의 에지 결과를 합쳐서 전체 에지 강도 계산하기 
magnitude_uint8 = cv.convertScaleAbs(magnitude) # 화면에 그리기 위해 계산된 강도 값을 8비트 이미지 포맷(uint8)으로 변환하기 


# 4. Matplotlib를 사용한 시각화
plt.figure(figsize=(10, 5)) # 전체 출력 창의 크기를 가로 10, 세로 5 비율로 설정하기

plt.subplot(1, 2, 1) # 창을 1행 2열로 쪼개고 그중 첫 번째 칸 선택하기
plt.imshow(img1_rgb) # 첫 번째 칸에 원본 컬러 이미지 띄우기
plt.title('Original Image') # 첫 번째 이미지 위에 제목 달기
plt.axis('off') # 지저분해 보이지 않게 x, y축 눈금선 숨기기

plt.subplot(1, 2, 2) # 두 번째 칸 선택하기
plt.imshow(magnitude_uint8, cmap='gray') # 두 번째 칸에 에지 강도 이미지를 흑백 톤으로 띄우기 
plt.title('Sobel Edge Magnitude') # 두 번째 이미지 위에 제목 달기
plt.axis('off') # 마찬가지로 축 눈금선 숨기기

plt.show() # 위에서 설정한 두 개의 이미지를 화면에 나란히 출력하기 

```


### 3. 핵심 코드
- gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) : 컬러 이미지에 포함된 색상 정보는 윤곽선 검출에 불필요하므로, 연산 속도 향상과 정확도를 위해 원본 이미지를 그레이스케일(흑백)로 변환.
- sobelx = cv.Sobel(gray1, cv.CV_64F, 1, 0, ksize=3) : 소벨 필터를 적용하여 x축 방향(1, 0)의 에지를 검출합니다. 음수 값 보존을 위해 반환 타입을 cv.CV_64F로 지정하고, 마스크 크기는 3x3(ksize=3)으로 설정
- sobely = cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=3) : 소벨 필터를 적용하여 y축 방향(0, 1)의 에지를 검출
- magnitude = cv.magnitude(sobelx, sobely) : 도출된 x축 방향의 에지 결과(sobelx)와 y축 방향의 에지 결과(sobely)를 결합하여 최종적인 에지 강도(크기)를 계산
- magnitude_uint8 = cv.convertScaleAbs(magnitude) : 계산된 실수형 에지 강도 이미지에 절댓값을 적용하고, 이를 모니터에 시각화할 수 있는 부호 없는 8비트 정수형(uint8)으로 변환



### 4. 실행 결과


<img width="1000" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/e1ed4cfd-d352-4f5f-b0fb-1d02bfbd68ab" />


---

## 🌀 문제 02 캐니 에지 및 허프 변환을 이용한 직선 검출



### 1. 주요 개념


### 2. 전체 코드

```Python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 1. 이미지 불러오기 및 변환
img2 = cv.imread('dabo.jpg') # 다보탑 이미지를 파일에서 읽어오기
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB) # 화면 출력을 위해 BGR 색상을 RGB로 변환하기
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) # 에지 검출 연산을 위해 이미지를 흑백(그레이스케일)으로 변환하기


# 2. 캐니 에지 검출 (threshold1=100, threshold2=200 설정)
edges = cv.Canny(gray2, 100, 200) # 캐니 알고리즘으로 에지(윤곽선) 맵 생성하기 (최소 임계값 100, 최대 임계값 200)


# 3. 허프 변환을 통한 직선 검출 (파라미터 조정 필요)
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=80, maxLineGap=10)


# 4. 검출된 직선 그리기
img2_lines = img2.copy() # 원본 이미지에 덧그리기 위해 복사본 만들기
if lines is not None: # 선이 하나라도 검출되었다면
    for line in lines: # 찾은 선들을 하나씩 꺼내서 반복 작업하기
        x1, y1, x2, y2 = line[0] # 선의 시작점(x1, y1)과 끝점(x2, y2) 좌표 가져오기
        cv.line(img2_lines, (x1, y1), (x2, y2), (0, 0, 255), 2) # 이미지에 빨간색(0, 0, 255)으로 두께 2짜리 선 그리기


# Matplotlib 시각화를 위해 RGB 변환
img2_lines_rgb = cv.cvtColor(img2_lines, cv.COLOR_BGR2RGB) # 선이 그려진 결과 이미지도 화면 출력을 위해 RGB로 변환


# 5. 시각화
plt.figure(figsize=(10, 5)) # 전체 출력 창의 크기를 가로 10, 세로 5 비율로 설정하기

plt.subplot(1, 2, 1) # 창을 1행 2열로 쪼개고 첫 번째 칸 선택하기
plt.imshow(img2_rgb) # 첫 번째 칸에 원본 컬러 이미지 띄우기
plt.title('Original Image') # 첫 번째 이미지 위에 제목 달기
plt.axis('off') # x, y축 눈금선 숨기기

plt.subplot(1, 2, 2) # 두 번째 칸 선택하기
plt.imshow(img2_lines_rgb) # 두 번째 칸에 붉은 직선이 덧그려진 결과 이미지 띄우기
plt.title('Hough Lines') # 두 번째 이미지 위에 제목 달기
plt.axis('off') # 눈금선 숨기기

plt.show() # 설정한 두 개의 이미지를 화면에 나란히 출력하기
```


### 3. 핵심 코드



### 4. 실행 결과


<img width="1000" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/24c494b4-b40a-4d39-bf8f-0d06f12ddd93" />


---

## 🌀 문제 03 GrabCut을 이용한 대화식 영역 분할 및 객체 추출



### 1. 주요 개념



### 2. 전체 코드

```Python
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 RGB 변환
img3 = cv.imread('coffee cup.jpg') # 실습용 커피잔 이미지를 파일에서 읽어오기
img3_rgb = cv.cvtColor(img3, cv.COLOR_BGR2RGB) # 화면 출력을 위해 BGR 색상을 RGB로 변환하기


# 2. GrabCut 초기화 및 모델 생성
mask = np.zeros(img3.shape[:2], np.uint8) # 원본 이미지와 같은 크기의 빈 마스크(검은색 도화지) 생성하기
bgdModel = np.zeros((1, 65), np.float64) # GrabCut 알고리즘이 사용할 배경 모델 초기화 (임시 공간)
fgdModel = np.zeros((1, 65), np.float64) # GrabCut 알고리즘이 사용할 전경(객체) 모델 초기화 (임시 공간)


# 3. 초기 사각형 영역 설정 (x, y, width, height)
height, width = img3.shape[:2] # 원본 이미지의 세로(높이)와 가로(너비) 크기 가져오기
rect = (50, 50, width-100, height-100) # 상하좌우 50픽셀씩 여백을 둔 사각형 영역(x, y, 가로길이, 세로길이) 만들기


# 4. 대화식 분할 수행 (GrabCut) 
# 우리가 지정한 사각형(rect)을 바탕으로 GrabCut 알고리즘을 5번 반복 실행하여 배경과 전경 분리하기
cv.grabCut(img3, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)


# 5. 마스크 처리 및 배경 제거
# 확실한 배경(0)이거나 예상되는 배경(2)인 부분은 0(검정)으로, 객체 부분은 1(흰색)로 마스크 값 변경하기
mask2 = np.where((mask==cv.GC_PR_BGD)|(mask==cv.GC_BGD), 0, 1).astype('uint8') 
img3_no_bg = img3 * mask2[:, :, np.newaxis] # 원본 이미지에 마스크를 곱해서 배경 부분은 까맣게 지우고 전경만 남기기


# 시각화를 위한 색상 변환
img3_no_bg_rgb = cv.cvtColor(img3_no_bg, cv.COLOR_BGR2RGB) # 배경이 지워진 결과 이미지도 화면 출력을 위해 RGB로 변환하기


# 6. 세 가지 결과 시각화
plt.figure(figsize=(15, 5)) # 전체 출력 창의 크기를 가로 15, 세로 5 비율로 넓게 설정하기

plt.subplot(1, 3, 1) # 창을 1행 3열로 쪼개고 첫 번째 칸 선택하기
plt.imshow(img3_rgb) # 원본 컬러 이미지 띄우기
plt.title('Original Image') # 제목 달기
plt.axis('off') # 눈금선 숨기기

plt.subplot(1, 3, 2) # 두 번째 칸 선택하기
plt.imshow(mask2, cmap='gray') # GrabCut 알고리즘이 찾아낸 흑백 마스크 이미지 띄우기
plt.title('Mask') # 제목 달기
plt.axis('off') # 눈금선 숨기기

plt.subplot(1, 3, 3) # 세 번째 칸 선택하기
plt.imshow(img3_no_bg_rgb) # 배경이 제거되고 객체(커피잔)만 덩그러니 남은 최종 결과 이미지 띄우기
plt.title('Background Removed') # 제목 달기
plt.axis('off') # 눈금선 숨기기

plt.show() # 세 개의 이미지를 화면에 나란히 출력하기
```


### 3. 핵심 코드



### 4. 실행 결과


<img width="1470" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/ca4e6566-a5af-404a-81f0-43c6739ee726" />


---
