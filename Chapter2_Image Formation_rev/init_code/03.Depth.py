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