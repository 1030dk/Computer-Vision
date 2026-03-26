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