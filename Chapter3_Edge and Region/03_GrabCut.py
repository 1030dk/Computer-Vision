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