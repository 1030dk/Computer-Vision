import cv2 as cv  # OpenCV 라이브러리 로드
import matplotlib.pyplot as plt  # 시각화(그래프/이미지) 라이브러리 로드
import numpy as np  # 배열 및 행렬 연산을 위한 넘파이 로드

# 1. 두 이미지 불러오기 (img1.jpg, img2.jpg 선택)
img_left = cv.imread('img1.jpg')  # 왼쪽(기준) 이미지 로드
img_right = cv.imread('img2.jpg')  # 오른쪽(변환 대상) 이미지 로드

gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)  # 왼쪽 이미지 흑백 변환
gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)  # 오른쪽 이미지 흑백 변환

# 2. SIFT 특징점 검출 및 매칭
sift = cv.SIFT_create()  # SIFT 객체 생성
kp1, des1 = sift.detectAndCompute(gray_left, None)  # 왼쪽 이미지 특징점/디스크립터 추출 
kp2, des2 = sift.detectAndCompute(gray_right, None)  # 오른쪽 이미지 특징점/디스크립터 추출 

bf = cv.BFMatcher()  # 매칭기 객체 생성 [cite: 54]
matches = bf.knnMatch(des1, des2, k=2)  # 최근접 이웃 2개(k=2)를 찾는 매칭 수행 

good_matches = []  # 우수한 매칭점만 담을 리스트
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 거리 비율이 0.7 미만인 훌륭한 매칭점만 선별
        good_matches.append(m)

# 3. 매칭 결과 이미지 생성
img_match_result = cv.drawMatches(
    img_left, kp1, img_right, kp2,  # 두 원본 이미지와 각 특징점들
    good_matches,  # 선별된 우수 매칭점 연결
    None, 
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 매칭 안 된 점은 생략
)

# 4. 호모그래피 계산 및 이미지 변환
if len(good_matches) >= 4:  # 호모그래피 계산에 필요한 최소 조건(4개) 확인
    # 핵심 수정 부분: 오른쪽 이미지를 왼쪽 이미지 기준으로 변환
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 오른쪽 이미지의 매칭점 좌표
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 왼쪽 이미지의 매칭점 좌표 (기준)

    # RANSAC을 이용해 이상점(Outlier)을 배제하며 호모그래피 행렬(H) 계산
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0) 

    # 파노라마 도화지 크기 설정
    h1, w1 = img_left.shape[:2]  # 왼쪽 이미지 세로, 가로 길이
    h2, w2 = img_right.shape[:2]  # 오른쪽 이미지 세로, 가로 길이
    panorama_w = w1 + w2  # 두 이미지 가로를 합친 넉넉한 도화지 너비 [cite: 61]
    panorama_h = max(h1, h2)  # 두 이미지 중 더 긴 세로 길이 적용 [cite: 61]

    # 오른쪽 이미지를 호모그래피 행렬(H)을 이용해 변환하여 넓은 도화지에 배치 [cite: 56]
    warped_img = cv.warpPerspective(img_right, H, (panorama_w, panorama_h))
    
    # 도화지의 왼쪽 빈 공간에 기준이 되는 원본 왼쪽 이미지를 그대로 덮어쓰기
    warped_img[0:h1, 0:w1] = img_left

    # 5. 결과 시각화
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 위아래(2행 1열)로 2개 이미지를 띄울 공간 생성
    
    axes[0].imshow(cv.cvtColor(img_match_result, cv.COLOR_BGR2RGB))  # 위쪽에 매칭 결과 출력
    axes[0].set_title('Matching Result (Ratio < 0.7)')  # 제목 설정
    axes[0].axis('off')  # 축 숨김
    
    axes[1].imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))  # 아래쪽에 정합된 파노라마 이미지 출력
    axes[1].set_title('Warped Image (Panorama Alignment)')  # 제목 설정
    axes[1].axis('off')  # 축 숨김
    
    plt.tight_layout()  # 여백 자동 조정
    plt.show()  # 화면에 띄우기

else:
    print("호모그래피를 계산하기 위한 충분한 매칭점을 찾지 못했습니다.")