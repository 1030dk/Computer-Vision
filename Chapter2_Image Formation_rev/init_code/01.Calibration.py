import cv2
import numpy as np
import glob
import sys
import os

# -----------------------------
# [초기 설정] 기본 파라미터
# -----------------------------
CHECKERBOARD = (9, 6)  # 가로 9, 세로 6 코너
square_size = 25.0     # 한 칸의 실제 크기 (25mm)

# 코너 정밀화 조건 (정확도 도달 혹은 최대 반복 시 종료)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 세계의 3D 좌표 (Object Points) 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # 실제 크기 반영

# 이미지 및 실제 좌표 데이터를 저장할 리스트
objpoints = [] # 3D 실제 좌표
imgpoints = [] # 2D 이미지 좌표

# -----------------------------
# [경로 탐색] 이미지 파일 불러오기
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, "..", "images", "calibration_images", "left*.jpg")
images = glob.glob(target_path)

if len(images) == 0:
    print("[경고] 이미지를 찾지 못했습니다! 경로를 다시 확인해 주세요.")
    sys.exit()

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    
    if img_size is None:
        img_size = gray.shape[::-1]  # 이미지 크기 저장

    # 코너 검출 수행
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너 검출에 성공한 이미지의 좌표만 추가
    if ret == True:
        objpoints.append(objp)
        
        # 코너 위치 정밀 보정 후 추가
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

if len(objpoints) == 0:
    print("[경고] 유효한 코너 데이터가 없어 캘리브레이션을 종료합니다.")
    sys.exit()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 카메라 내부 파라미터(K)와 왜곡 계수(dist) 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:
    test_img = cv2.imread(images[0])  # 테스트용 첫 번째 이미지
    h, w = test_img.shape[:2]
    
    # 최적화된 새 카메라 행렬 계산
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    
    # 원본 이미지의 왜곡 보정 적용
    undistorted_img = cv2.undistort(test_img, K, dist, None, newcameramtx)
    
    # 원본(좌)과 보정본(우)을 가로로 이어붙임
    combined_img = np.hstack((test_img, undistorted_img))
    
    # 창 크기 조절 가능하게 설정 후 출력
    cv2.namedWindow("Original(Left) vs Undistorted(Right)", cv2.WINDOW_NORMAL)
    cv2.imshow("Original(Left) vs Undistorted(Right)", combined_img)
    
    cv2.waitKey(0)           # 키 입력 대기
    cv2.destroyAllWindows()  # 모든 창 닫기