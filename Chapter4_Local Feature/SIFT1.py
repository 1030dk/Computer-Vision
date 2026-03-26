import sys  # 시스템 환경 확인용 (제출 시 삭제 가능)
print("현재 실행 중인 파이썬 경로:", sys.executable)  # 실행 중인 파이썬 인터프리터 경로 출력

import cv2 as cv  # OpenCV 라이브러리 로드
import matplotlib.pyplot as plt  # 시각화(그래프/이미지) 라이브러리 로드

# 1. 이미지 로드
img1 = cv.imread('mot_color70.jpg')  # 이미지 파일 읽기 (BGR 포맷)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # SIFT 처리를 위해 컬러 이미지를 흑백으로 변환

# 2. SIFT 객체 생성 및 특징점 제한
sift = cv.SIFT_create(nfeatures=500)  # SIFT 객체 생성 (가장 뚜렷한 특징점 500개만 추출)

# 3. 특징점 검출 및 특징 디스크립터 계산
kp1, des1 = sift.detectAndCompute(gray1, None)  # 흑백 이미지에서 특징점(kp1)과 디스크립터(des1) 계산

# 4. 특징점 시각화 (방향과 크기 표시)
img1_kp = cv.drawKeypoints(gray1, kp1, img1.copy(), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 원본 위에 특징점의 크기와 방향을 원과 선으로 시각화

# 5. Matplotlib을 이용한 결과 비교 출력
plt.figure(figsize=(12, 6))  # 가로 12, 세로 6 크기의 전체 도화지 생성

plt.subplot(1, 2, 1)  # 1행 2열로 화면을 나누고, 첫 번째(왼쪽) 영역 선택
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))  # 화면에 정상적인 색으로 보이도록 BGR을 RGB로 변환하여 출력
plt.title('Original Image')  # 왼쪽 이미지 제목 설정
plt.axis('off')  # 불필요한 X, Y축 눈금 숨기기

plt.subplot(1, 2, 2)  # 두 번째(오른쪽) 영역 선택
plt.imshow(cv.cvtColor(img1_kp, cv.COLOR_BGR2RGB))  # 특징점이 그려진 결과 이미지를 RGB로 변환하여 출력
plt.title('SIFT Keypoints (Rich)')  # 오른쪽 이미지 제목 설정
plt.axis('off')  # 축 눈금 숨기기

plt.tight_layout()  # 그림과 제목이 서로 겹치지 않도록 여백을 자동으로 예쁘게 조정
plt.show()  # 완성된 결과를 화면 창에 띄우기