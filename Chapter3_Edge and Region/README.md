## 🌀 문제 01 소벨 에지 검출 및 결과 시각화

![1111](https://github.com/user-attachments/assets/91f819ef-ed24-40e7-b4e1-ea5fe19a4b55)


### 1.주요 개념

- 에지(Edge)와 밝기 변화: 영상 처리에서 에지(윤곽선)는 픽셀의 밝기나 색상이 급격하게 변하는 경계면을 의미합니다. 컴퓨터는 이 밝기 변화량(미분값)이 큰 곳을 찾아 에지로 인식.
- 소벨 필터(Sobel Filter): 이미지의 특정 방향으로 미분 연산을 수행하여 에지를 검출하는 대표적인 마스크(필터)입니다.
    - x축 방향: 좌우의 밝기 변화를 측정하여 수직선(세로 방향 윤곽선)을 검출.
    - y축 방향: 상하의 밝기 변화를 측정하여 수평선(가로 방향 윤곽선)을 검출.
- 에지 강도(Magnitude): 독립적으로 계산된 x축 미분값과 y축 미분값을 피타고라스 정리($\sqrt{x^2 + y^2}$)를 통해 하나의 벡터 크기로 합산하는 과정(뚜렷한 윤곽석을 얻을 수 있음.)



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
- **`gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)`** : 컬러 이미지에 포함된 색상 정보는 윤곽선 검출에 불필요하므로, 연산 속도 향상과 정확도를 위해 원본 이미지를 그레이스케일(흑백)로 변환.
- **`sobelx = cv.Sobel(gray1, cv.CV_64F, 1, 0, ksize=3)`** : 소벨 필터를 적용하여 x축 방향(1, 0)의 에지를 검출합니다. 음수 값 보존을 위해 반환 타입을 cv.CV_64F로 지정하고, 마스크 크기는 3x3(ksize=3)으로 설정
- **`sobely = cv.Sobel(gray1, cv.CV_64F, 0, 1, ksize=3)`** : 소벨 필터를 적용하여 y축 방향(0, 1)의 에지를 검출
- **`magnitude = cv.magnitude(sobelx, sobely)`** : 도출된 x축 방향의 에지 결과(sobelx)와 y축 방향의 에지 결과(sobely)를 결합하여 최종적인 에지 강도(크기)를 계산
- **`magnitude_uint8 = cv.convertScaleAbs(magnitude)`** : 계산된 실수형 에지 강도 이미지에 절댓값을 적용하고, 이를 모니터에 시각화할 수 있는 부호 없는 8비트 정수형(uint8)으로 변환



### 4. 실행 결과


<img width="1000" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/e1ed4cfd-d352-4f5f-b0fb-1d02bfbd68ab" />


---

## 🌀 문제 02 캐니 에지 및 허프 변환을 이용한 직선 검출

![222](https://github.com/user-attachments/assets/c9ac4e94-391d-4785-b239-e6215a0f9929)



### 1. 주요 개념

- 캐니 에지 검출(Canny Edge Detection): 노이즈에 민감한 기존 에지 검출 알고리즘의 단점을 보완한 다단계 에지 검출 기법.
- 허프 변환(Hough Transform): 이미지 상의 에지 픽셀(x, y)들을 수학적인 파라미터 공간($r$, $\theta$)으로 변환하여 직선, 원 등의 기하학적 형태를 찾아내는 알고리즘.
    - 여러 에지 점들이 파라미터 공간에서 한 점으로 교차(투표, Voting)할 때, 그 점들을 이으면 하나의 직선이 된다는 원리를 이용 
- 확률적 허프 변환(Probabilistic Hough Transform): 영상의 모든 픽셀을 연산하여 속도가 느린 기본 허프 변환의 단점을 극복하기 위해 고안된 방법
    - 모든 점을 연산하지 않고 무작위로 필요한 만큼의 픽셀만 샘플링하여 빠르게 직선을 찾아냄. 


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

- **`edges = cv.Canny(gray2, 100, 200)`** : 레이스케일 이미지(gray2)를 입력받아 캐니 에지 맵을 생성합니다. 최소 임계값(100)과 최대 임계값(200)을 설정하여, 에지 강도가 200 이상인 확실한 경계선은 보존하고, 100~200 사이의 픽셀은 확실한 경계선과 연결되어 있을 때만 에지로 인정하여 불필요한 노이즈를 제거
- **`lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=80, maxLineGap=10)`** : 생성된 에지 맵(edges)을 바탕으로 확률적 허프 변환을 수행하여 직선을 검출합니다.  1은 거리 정밀도(1픽셀 단위), np.pi/180은 각도 정밀도(1도 단위)를 의미합니다. threshold는 직선으로 판별하기 위한 최소 교차점(투표수)을 뜻하며, minLineLength는 검출할 선분의 최소 길이, maxLineGap은 끊어진 선분을 하나의 선으로 간주할 최대 허용 간격을 의미합니다. 이 파라미터들을 튜닝하여 직선 검출 성능을 목표에 맞게 개선
- **`cv.line(img2_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)`** : 허프 변환을 통해 반환된 직선들의 집합(lines)에서 개별 선분의 시작점(x1, y1)과 끝점(x2, y2) 좌표를 추출한 뒤, 원본 이미지 복사본 위에 직선을 그립니다. 색상은 BGR 기준 빨간색인 (0, 0, 255)로, 선의 두께는 2로 설정하여 시각적으로 검출 결과를 명확히 표시



### 4. 실행 결과


<img width="1000" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/24c494b4-b40a-4d39-bf8f-0d06f12ddd93" />


---

## 🌀 문제 03 GrabCut을 이용한 대화식 영역 분할 및 객체 추출

![333](https://github.com/user-attachments/assets/7fc640bb-2159-4a52-959d-3e253cfa45ef)




### 1. 주요 개념
- 영역 분할(Image Segmentation): 이미지 내에서 의미 있는 객체(전경)와 배경을 픽셀 단위로 분리해 내는 컴퓨터 비전 기술.
- GrabCut 알고리즘: 사용자가 분리하고자 하는 객체 주변에 대략적인 사각형(Bounding Box)을 지정해 주면, 그 정보를 바탕으로 전경과 배경의 색상 분포(Gaussian Mixture Model)를 통계적으로 학습하여 객체의 외곽선을 정교하게 따내는 대화식 분할 알고리즘.
- 마스크(Mask): 이미지의 특정 픽셀을 남길지, 지울지 결정하는 일종의 필터입니다. GrabCut 알고리즘은 연산 과정에서 각 픽셀을 4가지 상태('확실한 배경', '아마도 배경', '확실한 전경', '아마도 전경') 중 하나로 분류하여 마스크에 기록.


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

- **`bgdModel = np.zeros((1, 65), np.float64) 및 fgdModel = np.zeros((1, 65), np.float64)`** : GrabCut 알고리즘이 배경(bgdModel)과 전경(fgdModel)의 색상 모델 정보를 학습하고 임시로 저장해 둘 빈 배열을 생성합니다. 내부 연산 규칙에 따라 반드시 크기가 (1, 65)인 float64 자료형 배열로 초기화해야 함.
- **`cv.grabCut(img3, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)`** : 설정된 사각형 영역(rect) 정보를 바탕으로 GrabCut 알고리즘을 5회 반복 실행하여 대화식 분할을 수행
    - cv.GC_INIT_WITH_RECT는 사용자가 초기 영역을 사각형으로 지정했음을 알고리즘에 알려주는 모드
- **`mask2 = np.where((mask==cv.GC_PR_BGD)|(mask==cv.GC_BGD), 0, 1).astype('uint8')`** : GrabCut 연산 결과로 채워진 마스크(mask)에서 '확실한 배경(cv.GC_BGD)'과 '아마도 배경(cv.GC_PR_BGD)'으로 판정된 픽셀 값은 0으로, 전경에 해당하는 나머지 값은 1로 변경합니다.
- **`img3_no_bg = img3 * mask2[:, :, np.newaxis]`** : 생성된 이진 마스크(mask2)의 채널을 컬러 이미지에 맞게 확장(np.newaxis)한 뒤, 원본 이미지 배열과 곱합니다. 0이 곱해진 배경 부분의 픽셀은 모두 검게 지워지고, 1이 곱해진 전경(커피잔) 픽셀만 남게 되어 최종적으로 배경이 제거된 이미지가 출력됨.



### 4. 실행 결과


<img width="1470" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/ca4e6566-a5af-404a-81f0-43c6739ee726" />


---
