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