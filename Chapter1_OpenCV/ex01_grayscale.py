import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # 배열 처리를 위한 NumPy 임포트

img = cv.imread('girl_laughing.jpg') # 원본 이미지 로드

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지를 흑백(그레이스케일)으로 변환

gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 가로 연결을 위해 흑백 이미지를 3채널로 임시 변환

result = np.hstack((img, gray_3ch)) # 원본과 흑백 이미지를 가로로 연결

cv.imshow('Original and Grayscale', result) # 결과 이미지를 화면에 출력
cv.waitKey(0) # 키보드 입력 대기 (아무 키나 누르면 다음으로 넘어감)
cv.destroyAllWindows() # 열려있는 모든 창 닫기