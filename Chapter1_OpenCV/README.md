## 🌀문제 01 이미지 불러오기 및 그레이스케일 변환


<전체코드>
```Python

import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # 배열 처리를 위한 NumPy 임포트

img = cv.imread('girl_laughing.jpg') # 원본 이미지 로드

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지를 흑백(그레이스케일)으로 변환

gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 가로 연결을 위해 흑백 이미지를 3채널로 임시 변환

result = np.hstack((img, gray_3ch)) # 원본과 흑백 이미지를 가로로 연결

cv.imshow('Original and Grayscale', result) # 결과 이미지를 화면에 출력
cv.waitKey(0) # 키보드 입력 대기 (아무 키나 누르면 다음으로 넘어감)
cv.destroyAllWindows() # 열려있는 모든 창 닫기
```

<실행결과>

![01result](https://github.com/user-attachments/assets/fc2dd382-204e-494d-8a59-469b98c450e7)

---
## 🌀문제 02 페인팅 붓 크기 조절 기능 추가

<전체코드>
```Python
import cv2 as cv # OpenCV 라이브러리 임포트

img = cv.imread('soccer.jpg') # 배경 이미지 로드
brush_size = 5 # 초기 붓 크기 설정
drawing = False # 그리기 상태 플래그
color = (255, 0, 0) # 초기 붓 색상 (파란색)

def paint(event, x, y, flags, param): # 마우스 이벤트 처리 콜백 함수
    global drawing, color, brush_size, img # 전역 변수 사용 선언
    
    if event == cv.EVENT_LBUTTONDOWN: # 좌클릭 시
        drawing = True # 그리기 상태 활성화
        color = (255, 0, 0) # 파란색 설정
        cv.circle(img, (x, y), brush_size, color, -1) # 원 그리기
    elif event == cv.EVENT_RBUTTONDOWN: # 우클릭 시
        drawing = True # 그리기 상태 활성화
        color = (0, 0, 255) # 빨간색 설정
        cv.circle(img, (x, y), brush_size, color, -1) # 원 그리기
    elif event == cv.EVENT_MOUSEMOVE: # 마우스 이동 시
        if drawing: # 그리기 상태라면
            cv.circle(img, (x, y), brush_size, color, -1) # 이동 궤적을 따라 연속해서 원 그리기
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP: # 클릭 해제 시
        drawing = False # 그리기 상태 비활성화

cv.namedWindow('Paint') # 그림을 그릴 창 생성
cv.setMouseCallback('Paint', paint) # 마우스 콜백 함수 등록

while True: # 무한 루프 시작
    cv.imshow('Paint', img) # 이미지 화면 출력
    key = cv.waitKey(1) & 0xFF # 키보드 입력 대기
    
    if key == ord('q') or key == ord('Q'): # q 키 입력 시
        break # 루프 종료
    elif key == ord('+') or key == ord('='): # + 키 입력 시
        brush_size = min(15, brush_size + 1) # 붓 크기 증가 (최대 15)
        print(f"현재 붓 크기: {brush_size}") # 변경된 크기 출력
    elif key == ord('-'): # - 키 입력 시
        brush_size = max(1, brush_size - 1) # 붓 크기 감소 (최소 1)
        print(f"현재 붓 크기: {brush_size}") # 변경된 크기 출력

cv.destroyAllWindows() # 열려있는 모든 창 닫기
```

<실행 결과>

![02result](https://github.com/user-attachments/assets/b9b02936-db1e-47d8-8217-cdba799c95d4)


---
## 🌀문제 03 마우스로 영역 선택 및 ROI(관심영역) 추출

<전체코드>
```Python
import cv2 as cv # OpenCV 라이브러리 임포트

img = cv.imread('girl_laughing.jpg') # 원본 이미지 로드
clone = img.copy() # 화면 표시용 이미지 복사본 생성
roi = None # ROI(관심 영역) 데이터 저장 변수 초기화
drawing = False # 드래그 상태 플래그
ix, iy = -1, -1 # 드래그 시작 좌표 초기화

def draw_roi(event, x, y, flags, param): # 마우스 이벤트 처리 콜백 함수
    global ix, iy, drawing, clone, img, roi # 전역 변수 사용 선언

    if event == cv.EVENT_LBUTTONDOWN: # 좌클릭 시
        drawing = True # 드래그 시작
        ix, iy = x, y # 시작 좌표 저장
    elif event == cv.EVENT_MOUSEMOVE: # 마우스 이동 시
        if drawing: # 드래그 중이라면
            clone = img.copy() # 이전 사각형 잔상 지우기
            cv.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2) # 초록색 사각형 시각화
    elif event == cv.EVENT_LBUTTONUP: # 클릭 해제 시
        drawing = False # 드래그 종료
        cv.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2) # 최종 사각형 확정
        
        x1, x2 = min(ix, x), max(ix, x) # X 좌표 정렬 (역방향 드래그 대비)
        y1, y2 = min(iy, y), max(iy, y) # Y 좌표 정렬 (역방향 드래그 대비)
        
        if x2 - x1 > 0 and y2 - y1 > 0: # 유효한 크기의 영역일 경우
            roi = img[y1:y2, x1:x2] # ROI 이미지 잘라내기
            cv.imshow('ROI', roi) # 잘라낸 ROI를 전용 창에 출력

cv.namedWindow('Image') # 메인 이미지 창 생성
cv.setMouseCallback('Image', draw_roi) # 마우스 콜백 함수 등록

while True: # 무한 루프 시작
    cv.imshow('Image', clone) # 이미지 화면 출력
    key = cv.waitKey(1) & 0xFF # 키보드 입력 대기
    
    if key == ord('r') or key == ord('R'): # r 키 입력 시
        clone = img.copy() # 화면의 사각형 지우기 (초기화)
        roi = None # 저장된 ROI 데이터 초기화
        if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 0: # ROI 창이 열려있다면
            cv.destroyWindow('ROI') # ROI 창 닫기
        print("영역 선택 리셋 완료") # 리셋 안내 메시지 출력
    elif key == ord('s') or key == ord('S'): # s 키 입력 시
        if roi is not None: # 선택된 ROI가 존재한다면
            cv.imwrite('saved_roi.jpg', roi) # 영역을 이미지 파일로 저장
            print("saved_roi.jpg 저장 완료") # 저장 완료 메시지 출력
        else: # 선택된 ROI가 없다면
            print("선택된 영역이 없습니다") # 경고 메시지 출력
    elif key == ord('q') or key == ord('Q') or key == 27: # q 또는 ESC 키 입력 시
        break # 루프 종료

cv.destroyAllWindows() # 열려있는 모든 창 닫기
```

<실행결과>

![03result](https://github.com/user-attachments/assets/d2b9abc4-c31e-4c1a-adea-146437894322)


---
