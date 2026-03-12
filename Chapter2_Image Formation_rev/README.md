## 🌀 문제 01 체크보드 기반 카메라 캘리브레이션

### 1. 주요 개념

카메라 캘리브레이션은 현실 세계의 3D 좌표와 카메라가 찍은 2D 이미지 좌표 사이의 관계를 찾아내어 카메라의 특성을 파악하고 왜곡을 펴는 작업입니다.
• 체크보드 (Checkerboard)
  - 카메라 캘리브레이션을 위해 사용하는 흑백 격자 패턴입니다. 코너(Corner)가 규칙적인 격자 구조를 띠고 있습니다
  - 실제 좌표를 정확히 알고 있는 패턴이기 때문에 이미지에서 검출된 코너와 실제 좌표의 대응 관계를 이용해 카메라의 내부 파라미터와 렌즈 왜곡을 계산할 수 있습니다.

• 카메라 내부 파라미터 행렬 (Camera Matrix, $K$)
  - <img width="168" height="84" alt="image" src="https://github.com/user-attachments/assets/c56561f0-0601-4178-906e-feee05e0e867" />
  - $f$ (Focal Length, 초점 거리): 카메라가 얼마나 이미지를 확대해서 찍는지를 픽셀 단위로 나타낸 값입니다.
  - $p$ (Principal Point, 주점): 렌즈 중심이 이미지에 찍히는 좌표입니다.


• 왜곡 계수 (Distortion Coefficients)
  - 행렬 형태: [k1, k2, p1, p2, k3]
  - 방사 왜곡 (Radial Distortion - $k_1, k_2, k_3$): 렌즈의 굴절 때문에 렌즈 중심에서 멀어질수록 직선이 휘어지는 현상입니다.
  - 접선 왜곡 (Tangential Distortion - $p_1, p_2$): 카메라 렌즈와 이미지 센서(필름 역할)가 완전히 평행하지 않을 때 발생하는 왜곡입니다.

### 2. 전체 코드

```Python
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
```


### 3. 핵심 코드
• cv2.calibrateCamera(objpoints, imgpoints, ...): 체크보드의 실제 3D 좌표와 검출된 2D 이미지 좌표를 이용하여 카메라 파라미터를 계산합니다. 카메라의 내부 행렬(Camera Matrix, 초점거리 및 주점)과 렌즈의 왜곡 계수(Distortion Coefficients)를 한 번에 반환합니다.

• undistorted_img = cv2.undistort(test_img, K, dist, None, newcameramtx): 앞서 구한 카메라 내부 행렬(K)과 왜곡 계수(dist)를 적용하여, 렌즈 굴절 등으로 인해 둥글게 휘어진 원본 이미지의 왜곡을 펴고 보정합니다



### 4. 실행 결과
<img width="1288" height="751" alt="01result" src="https://github.com/user-attachments/assets/db773ebc-e4b5-4792-af8f-4c17e6ee9a08" />

---

## 🌀 문제 02 이미지 Rotation & Transformation

### 1. 주요 개념
• 아핀 변환 (Affine Transformation): 선의 평행성을 유지하면서 이미지를 기하학적으로 변형(회전, 확대/축소, 평행이동)하는 컴퓨터 비전 기법입니다. 연산을 위해 2x3 크기의 변환 행렬을 사용합니다.

• 회전 및 크기 조절 (Rotation & Scaling): 특정 중심점을 기준으로 이미지를 주어진 각도만큼 회전시키고 크기를 비율에 맞게 조절합니다.

• 평행 이동 (Translation): 변환 행렬의 위치 이동 값을 수정하여 이미지를 x축(가로)과 y축(세로) 방향으로 원하는 픽셀만큼 이동시킵니다.

### 2. 전체 코드

```Python
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
```


### 3. 핵심 코드

• cv2.getRotationMatrix2D(center, angle, scale)`: 회전 중심점(`center`), 회전 각도(`angle`), 크기 조절 비율(`scale`)을 입력받아 2x3 크기의 변환 행렬을 생성합니다. (각도가 양수이면 반시계 방향 회전)

• v2.warpAffine(img, M, dsize): 최종적으로 완성된 아핀 변환 행렬(M)을 원본 이미지(img)에 적용하여 이미지를 변형합니다. dsize는 결과물로 나올 이미지의 크기(너비, 높이)입니다.


### 4. 실행 결과
![02result](https://github.com/user-attachments/assets/a5363ac7-c162-4941-b9f6-3800967a577d)

---

## 🌀 문제 03 Stereo Disparity 기반 Depth 추정

### 1. 주요 개념

• 스테레오 비전 (Stereo Vision): 사람의 두 눈이 거리를 가늠하는 원리를 모방한 컴퓨터 비전 기법입니다. 약간 떨어진 두 위치에서 촬영된 좌/우 이미지(Left/Right Image)를 비교해 물체의 깊이(거리)를 알아냅니다.

• 시차 (Disparity, $d$): 좌측 이미지와 우측 이미지에서 동일한 물체가 위치한 픽셀 좌표의 차이(이동량)를 의미합니다. 물체가 카메라와 가까울수록 시차가 크게 나타나고, 멀수록 시차가 작게 나타납니다.

• 깊이 (Depth, $Z$) 계산 공식: <img width="86" height="39" alt="image" src="https://github.com/user-attachments/assets/692fe095-a35b-44df-b5a8-e37aadcb98a8" />
  - $f$: 카메라의 초점 거리 (Focal Length)
  - $B$: 두 카메라 사이의 물리적 거리 (Baseline)
  - $d$: 계산된 시차 (Disparity)

### 2. 전체 코드

```Python
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# -----------------------------
# [초기 설정] 출력 폴더 및 경로 
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = Path(os.path.join(current_dir, "outputs"))
output_dir.mkdir(parents=True, exist_ok=True)  # outputs 폴더가 없으면 생성

# 상위 폴더(..)의 images 폴더에서 좌우 이미지 경로 지정
left_path = os.path.join(current_dir, "..", "images", "left.png")
right_path = os.path.join(current_dir, "..", "images", "right.png")

# 이미지 읽기
left_color = cv2.imread(left_path)
right_color = cv2.imread(right_path)

# 예외 처리: 이미지를 찾지 못하면 프로그램 종료
if left_color is None or right_color is None:
    print(f"[경고] 이미지를 찾지 못했습니다!\n경로(왼쪽): {left_path}")
    sys.exit()

# -----------------------------
# [파라미터 및 ROI 설정]
# -----------------------------
f = 700.0  # 초점 거리 (focal length)
B = 0.12   # 두 카메라 사이의 거리 (baseline)

# 관심 영역(ROI) 픽셀 좌표 (x, y, 가로, 세로)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 흑백(Grayscale) 이미지로 변환 (Stereo 매칭용)
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity(시차) 계산
# -----------------------------
# StereoBM 알고리즘으로 Disparity Map 계산
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_int = stereo.compute(left_gray, right_gray)

# 반환된 정수값을 실수로 변환 후 16으로 나누어 실제 Disparity 값 획득
disparity = disparity_int.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth(깊이) 계산
# -----------------------------
valid_mask = disparity > 0  # Disparity가 0보다 큰 유효 픽셀만 추출
depth_map = np.zeros_like(disparity, dtype=np.float32)

# Z = (f * B) / d 공식을 이용하여 유효 픽셀에만 깊이 계산 적용
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 Disparity / Depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # 각 ROI 영역만 잘라내기
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_mask = valid_mask[y:y+h, x:x+w]
    
    # 유효한 픽셀이 있으면 평균값 계산, 없으면 0 처리
    if np.any(roi_mask):
        avg_disp = np.mean(roi_disp[roi_mask])
        avg_depth = np.mean(roi_depth[roi_mask])
    else:
        avg_disp = 0.0
        avg_depth = 0.0
        
    results[name] = {"disparity": avg_disp, "depth": avg_depth}

# -----------------------------
# 4. 결과 출력 및 해석
# -----------------------------
print("=== ROI별 평균 Disparity 및 Depth ===")
for name, data in results.items():
    print(f"[{name}] 평균 Disparity: {data['disparity']:.2f} px, 평균 Depth: {data['depth']:.3f} m")

# Disparity가 클수록 가까운 물체, 작을수록 먼 물체
closest_roi = max(results, key=lambda k: results[k]['disparity'])
farthest_roi = min(results, key=lambda k: results[k]['disparity'] if results[k]['disparity'] > 0 else float('inf'))

print("\n=== 실행결과: 가장 가까운 ROI, 가장 먼 ROI ===")
print(f"가장 가까운 ROI: {closest_roi}")
print(f"가장 먼 ROI: {farthest_roi}")

# -----------------------------
# 5. Disparity Map 시각화 준비 (컬러 매핑)
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan  # 무효 픽셀은 NaN 처리

if np.all(np.isnan(disp_tmp)):
    print("[오류] 유효한 disparity 값이 없습니다.")
    sys.exit()

# 화면 표시를 위해 상/하위 5% 값을 기준으로 0~1 사이로 정규화
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

# 0~255 범위의 8비트 이미지로 변환
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# 제트(JET) 컬러맵 적용 (가까울수록 빨강, 멀수록 파랑)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. 원본(Left) 이미지에만 ROI 표시
# -----------------------------
left_vis = left_color.copy()

for name, (x, y, w, h) in rois.items():
    # 초록색 사각형과 텍스트로 관심 영역(ROI) 그리기
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 7. 하나의 창으로 합쳐서 출력
# -----------------------------
# 원본 이미지(좌)와 Disparity Map(우)을 가로로 이어 붙임
combined_img = np.hstack((left_vis, disparity_color))

# 창 크기를 마우스로 조절할 수 있게 설정
cv2.namedWindow("Original(Left) vs Disparity Map(Right)", cv2.WINDOW_NORMAL)
cv2.imshow("Original(Left) vs Disparity Map(Right)", combined_img)

cv2.waitKey(0)           # 키보드 입력 대기
cv2.destroyAllWindows()  # 창 모두 닫기
```


### 3. 핵심 코드

• cv2.StereoBM_create(numDisparities, blockSize): 스테레오 블록 매칭(Block Matching) 알고리즘 객체를 생성합니다. 탐색할 최대 시차 범위(numDisparities)와 매칭에 사용할 블록의 크기(blockSize)를 설정하는 역할을 합니다.

• stereo.compute(left_gray, right_gray): 흑백으로 변환된 좌/우 이미지를 입력하여 시차 맵(Disparity Map)을 계산합니다. 여기서 OpenCV 알고리즘이 반환하는 결과는 원래 값에서 16배 스케일링된 정수형 데이터입니다.

• 실제 Disparity 변환 연산: disparity_int.astype(np.float32) / 16.0: OpenCV가 반환한 16배 스케일링된 정수형 데이터를 실제 시차 값으로 사용하기 위해, 실수형(float32)으로 변환한 뒤 16으로 나누어 원래 수치로 되돌리는 필수 연산입니다.

• Depth 마스킹 및 계산: depth_map[valid_mask] = (f * B) / disparity[valid_mask]
시차가 0 이하인 부분(매칭 실패 또는 무한대 거리)을 제외하고, 유효한 픽셀에 대해서만 마스크(valid_mask)를 씌워 깊이 추정 공식($Z = \frac{f \times B}{d}$)을 일괄 적용해 최종 깊이(Depth) 맵을 만듭니다.

### 4. 실행 결과
<img width="900" height="555" alt="03result" src="https://github.com/user-attachments/assets/625139a1-da7a-4a9f-9aa2-9fbeb35871bf" />


---
