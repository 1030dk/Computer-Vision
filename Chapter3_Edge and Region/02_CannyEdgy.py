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