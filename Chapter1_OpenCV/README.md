## 문제 01 이미지 불러오기 및 그레이스케일 변환

₩₩₩
Python

import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # 배열 처리를 위한 NumPy 임포트

img = cv.imread('girl_laughing.jpg') # 원본 이미지 로드

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지를 흑백(그레이스케일)으로 변환

gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 가로 연결을 위해 흑백 이미지를 3채널로 임시 변환

result = np.hstack((img, gray_3ch)) # 원본과 흑백 이미지를 가로로 연결

cv.imshow('Original and Grayscale', result) # 결과 이미지를 화면에 출력
cv.waitKey(0) # 키보드 입력 대기 (아무 키나 누르면 다음으로 넘어감)
cv.destroyAllWindows() # 열려있는 모든 창 닫기
₩₩₩

<실행결과>
![01result](https://github.com/user-attachments/assets/fc2dd382-204e-494d-8a59-469b98c450e7)

---
## 문제 02 페인팅 붓 크기 조절 기능 추가


<실행결과>
![02result](https://github.com/user-attachments/assets/b9b02936-db1e-47d8-8217-cdba799c95d4)


---
## 문제 03 마우스로 영역 선택 및 ROI(관심영역) 추출

<실행결과>
![03result](https://github.com/user-attachments/assets/d2b9abc4-c31e-4c1a-adea-146437894322)


---
