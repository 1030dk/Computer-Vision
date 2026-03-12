import cv2
import numpy as np
import os
import sys

# -----------------------------
# [경로 탐색] 절대 경로 설정
# -----------------------------
# 현재 파일 위치를 기준으로 상위 폴더의 images/rose.png 경로 지정
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "..", "images", "rose.png") 

# 이미지 읽기
img = cv2.imread(image_path)

# 예외 처리: 이미지를 찾지 못하면 종료
if img is None:
    print(f"[경고] 이미지를 찾지 못했습니다!\n경로: {image_path}")
    sys.exit()

# -----------------------------
# 1. 이미지 중심점 계산
# -----------------------------
h, w = img.shape[:2]         # 이미지 높이(h), 너비(w)
center = (w / 2.0, h / 2.0)  # 중심 좌표 (x, y) 계산

# -----------------------------
# 2. 회전 및 크기 조절 행렬 생성
# -----------------------------
angle = 30.0  # +30도 회전
scale = 0.8   # 0.8배 축소
M = cv2.getRotationMatrix2D(center, angle, scale) # 회전 변환 행렬 생성

# -----------------------------
# 3. 평행이동 추가 적용
# -----------------------------
tx = 80.0   # x축 +80px 이동
ty = -40.0  # y축 -40px 이동

M[0, 2] += tx  # 행렬에 x축 평행이동 반영
M[1, 2] += ty  # 행렬에 y축 평행이동 반영

# -----------------------------
# 4. 아핀 변환 적용
# -----------------------------
# 통합된 변환 행렬(M)을 이미지에 한 번에 적용
transformed_img = cv2.warpAffine(img, M, (w, h))

# -----------------------------
# 5. 결과 출력
# -----------------------------
# 원본과 변환된 이미지를 가로로 나란히 이어붙임
combined_img = np.hstack((img, transformed_img))

# 창 크기 조절 가능하도록 설정 후 이미지 출력
window_name = "Original(Left) vs Transformed(Right)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, combined_img)

cv2.waitKey(0)           # 아무 키나 누를 때까지 대기
cv2.destroyAllWindows()  # 모든 창 닫기