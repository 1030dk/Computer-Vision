## 🌀 문제 01 SIFT를 이용한 특징점 검출 및 시각화


![11](https://github.com/user-attachments/assets/03226fca-8e99-4a36-a30f-259be0657675)


### 1.주요 개념

- SIFT(Scale-Invariant Feature Transform): 이미지의 크기(Scale)가 커지거나 작아져도, 혹은 이미지가 회전하거나 조명이 바뀌어도 변하지 않는 강력한 특징점을 추출하는 알고리즘.
- 특징점(Keypoints - kp): 이미지 내에서 코너나 모서리처럼 주변과 뚜렷하게 구별되어 추적하기 쉬운 좌표(x, y)
- 디스크립터(Descriptors - des): 추출된 특징점 주변의 픽셀들이 어떤 방향과 크기로 변화하는지를 계산하여 숫자로 나타냄.



### 2. 전체 코드

```Python
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

```


### 3. 핵심 코드
- **`sift = cv.SIFT_create(nfeatures=500)`** : cv.SIFT_create()를 사용하여 SIFT 객체를 생성합니다.
- **`kp1, des1 = sift.detectAndCompute(gray1, None)`** : 함수에 흑백 이미지(gray1)를 넣어 특징점(kp1)과 디스크립터(des1)를 한 번에 계산합니다. 컴퓨터 비전 연산의 효율성을 위해 보통 원본 컬러 이미지가 아닌 흑백 이미지를 사용.
- **`img1_kp = cv.drawKeypoints(
    gray1, kp1, img1.copy(), 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)`** : 검출된 특징점을 이미지 위에 시각적으로 그려줍니다.




### 4. 실행 결과
<img width="1200" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/f3ce8701-e47a-4af2-a6ed-9ea02f7c019f" />

---

## 🌀 문제 02 SIFT를 이용한 두 영상 간 특징점 매칭


![22](https://github.com/user-attachments/assets/c6512bb4-312b-4b39-a5a4-464000b62dd0)


### 1. 주요 개념

- 특징점 매칭(Feature Matching): 두 개의 다른 이미지(예: 약간 다른 각도에서 찍은 거리 사진)에서 각각 추출한 SIFT 디스크립터(숫자 배열)들의 거리를 계산하여, 서로 가장 비슷하게 생긴 짝을 찾아 연결하는 과정.
- KNN 매칭(K-Nearest Neighbors): 한 이미지의 특징점과 가장 비슷한 점을 다른 이미지에서 딱 1개만 찾는 것이 아니라, 가장 비슷한 점(1순위)과 두 번째로 비슷한 점(2순위)까지 총 $k$개(이 실습에서는 $k=2$)를 찾는 방법.
- 최근접 이웃 거리 비율 (Ratio Test): 1순위 매칭점과 2순위 매칭점의 거리를 비교하여, 1순위가 2순위보다 확실하게(비율적으로) 더 가까울 때만 '진짜 매칭'으로 인정하여 매칭 정확도를 획기적으로 높입.


### 2. 전체 코드

```Python
import cv2 as cv  # OpenCV 라이브러리 로드
import matplotlib.pyplot as plt  # 시각화(그래프/이미지) 라이브러리 로드

# 1. 두 이미지 불러오기
img1 = cv.imread('mot_color70.jpg')  # 첫 번째 원본 이미지 읽기 (BGR 포맷)
img2 = cv.imread('mot_color83.jpg')  # 두 번째 대상 이미지 읽기 (BGR 포맷)

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 첫 번째 이미지를 흑백으로 변환 (특징점 추출용)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # 두 번째 이미지를 흑백으로 변환

# 2. SIFT 특징점 추출
sift = cv.SIFT_create()  # SIFT 특징점 추출기 객체 생성
kp1, des1 = sift.detectAndCompute(gray1, None)  # 첫 번째 이미지의 특징점(kp1)과 디스크립터(des1) 계산
kp2, des2 = sift.detectAndCompute(gray2, None)  # 두 번째 이미지의 특징점(kp2)과 디스크립터(des2) 계산

# 3. BFMatcher 생성 및 knnMatch 적용
bf = cv.BFMatcher()  # 브루트 포스(Brute-Force) 매칭기 객체 생성
matches = bf.knnMatch(des1, des2, k=2)  # 각 특징점마다 가장 거리가 가까운 2개의 이웃(k=2)을 찾아 매칭

# 4. 최근접 이웃 거리 비율(Ratio Test) 적용
good_matches = []  # 정확하게 매칭된 결과만 담을 빈 리스트 생성
for m, n in matches:  # m: 가장 가까운 매칭점, n: 두 번째로 가까운 매칭점
    if m.distance < 0.75 * n.distance:  # 첫 번째 매칭점이 두 번째보다 확실히(0.75배 이하) 더 가까울 때만
        good_matches.append([m])  # 좋은 매칭(오류가 적은 매칭)으로 인정하여 리스트에 추가

# 5. 매칭 결과 시각화
img_matches = cv.drawMatchesKnn(
    img1, kp1, img2, kp2,  # 매칭할 두 원본 이미지와 각각의 특징점들
    good_matches,  # 위에서 걸러낸 훌륭한 매칭점들만 선으로 연결
    None,  # 출력할 결과 이미지 객체 (여기선 새로 생성하므로 None)
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 매칭되지 않고 혼자 남은 특징점들은 화면에 그리지 않음
)

# 6. Matplotlib 출력
plt.figure(figsize=(15, 6))  # 가로 15, 세로 6 크기의 넓은 도화지 생성
plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))  # 색상이 정상적으로 보이도록 BGR을 RGB로 변환하여 출력
plt.title('SIFT Feature Matching (KNN + Ratio Test)')  # 결과 이미지의 제목 달기
plt.axis('off')  # 불필요한 X, Y축 눈금 숨기기
plt.show()  # 완성된 매칭 결과 창 띄우기
```


### 3. 핵심 코드

- **`bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)`** : 브루트 포스 매칭기 객체를 생성합니다. 브루트 포스는 첫 번째 이미지의 모든 특징점을 두 번째 이미지의 모든 특징점과 일일이 다 비교하고 특징점당 가장 유사한 이웃을 2개씩 찾아 matches 변수에 저장.
- **`good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])`** : m은 1순위 매칭점, n은 2순위 매칭점, 1순위가 2순위보다 압도적으로 비슷할 때만 good_matches 리스트에 합격시켜 담아 잘못된 매칭 걸러냄.
- **`img_matches = cv.drawMatchesKnn(
    img1, kp1, img2, kp2, 
    good_matches, 
    None, 
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)`** : good_matches들만 두 이미지 사이에 선으로 이어 시각화, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS 옵션을 주어 짝을 찾지 못하고 혼자 남은 특징점들은 화면에 그리지 않아 결과물을 깔끔하게 만들어줌.



### 4. 실행 결과

<img width="1470" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/8686f8e0-a293-4f91-92af-67f485dd9556" />



---

## 🌀 문제 03 호모그래피를 이용한 이미지 정합 (Image Alignment)


![33](https://github.com/user-attachments/assets/95411969-a39e-431e-8418-e1773787f2e3)



### 1. 주요 개념
- 호모그래피 (Homography): 두 평면 사이의 원근 변환(투영 변환) 관계를 나타내는 3x3 행렬입니다. 한 카메라로 찍은 사진을 다른 위치나 각도에서 찍은 것처럼 시점을 기하학적으로 변환해줌.
- RANSAC (RANdom SAmple Consensus): 무작위로 샘플을 뽑아 모델을 예측하고, 그 예측에 가장 잘 들어맞는 정상 데이터(Inlier)가 많은 모델을 최종 선택하는 강력한 알고리즘
- 이미지 정합 및 스티칭 (Image Alignment & Stitching): 계산된 호모그래피 행렬을 바탕으로 한 이미지를 변형(Warping)하여 다른 이미지와 시점과 원근감을 맞춘 뒤, 하나의 큰 도화지에 나란히 이어 붙이는 과정(파노라마)


### 2. 전체 코드

```Python
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
```


### 3. 핵심 코드

- **`H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)`** : cv.findHomography() 함수를 사용하여 두 이미지의 특징점 좌표들(src_pts, dst_pts)을 완벽하게 포개어주는 변환 행렬 H를 계산, cv.RANSAC 옵션 을 사용하여 오차 허용 범위 설정
- **`warped_img = cv.warpPerspective(img_right, H, (panorama_w, panorama_h))
warped_img[0:h1, 0:w1] = img_left`** : 함수에 오른쪽 이미지(img_right)와 아까 구한 변환 행렬 H, 그리고 넉넉한 캔버스 크기를 넣어줍니다. 그러면 오른쪽 이미지가 왼쪽 이미지의 시점에 맞게 자연스럽게 회전하거나 찌그러지며 넓은 도화지 우측에 제자리를 찾아감



### 4. 실행 결과

<img width="1470" height="776" alt="Figure_3" src="https://github.com/user-attachments/assets/873f0971-f46a-46e6-bbaa-ad25f3f62d7f" />



---
