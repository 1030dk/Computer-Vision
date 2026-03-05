## 🌀 문제 01 이미지 불러오기 및 그레이스케일 변환

### 1. 주요 개념

![11](https://github.com/user-attachments/assets/7ced5792-ab65-4926-87fd-f9580a90f5a4)

컴퓨터는 이미지를 단순한 그림이 아니라 숫자들의 거대한 격자판(행렬)으로 인식.

• 픽셀(Pixel)과 채널: 이미지는 수많은 점(픽셀)으로 이루어져 있으며, 컬러 이미지는 보통 빨강(R), 초록(G), 파랑(B) 세 가지 색상 정보(채널)를 가짐.

• BGR 색상 공간: 일반적인 이미지 형식은 RGB 순서지만, OpenCV는 기본적으로 BGR(파랑, 초록, 빨강) 순서로 데이터를 읽음.

• 그레이스케일(Grayscale) 변환: 컬러 정보를 버리고 밝기 정보(0~255)만 남기는 과정입니다. 채널이 3개에서 1개로 줄어들기 때문에 컴퓨터가 연산해야 할 데이터양이 1/3로 감소하여 영상 처리의 기초 단계로 중요함.

### 2. 전체 코드

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
### 3. 핵심 코드

• img = cv.imread('girl_laughing.jpg'): 디스크에서 이미지 파일을 읽어 메모리(Numpy 행렬)에 적재.

• cv.cvtColor(img, cv.COLOR_BGR2GRAY): 컬러 이미지(3채널)를 흑백 이미지(1채널)로 변환합니다. 컴퓨터 비전에서 연산량을 줄이기 위해 자주 사용하는 전처리 단계.

• np.hstack((img1, img2)): 두 행렬을 가로로 연결합니다. 이때 두 이미지의 높이와 채널 수가 같아야 하므로, 흑백 이미지를 다시 3채널로 변환해주는 과정이 중요.

### 4. 실행 결과

![01result](https://github.com/user-attachments/assets/fc2dd382-204e-494d-8a59-469b98c450e7)

---
## 🌀 문제 02 페인팅 붓 크기 조절 기능 추가

### 1. 주요 개념

<img width="640" height="340" alt="22" src="https://github.com/user-attachments/assets/81a3c021-2c72-4226-978f-98352aa7a19b" />

사용자가 마우스를 움직이거나 클릭할 때 프로그램이 즉각 반응하게 만드는 기술.

• 이벤트(Event): 마우스 클릭, 드래그, 키보드 누름 등 사용자가 시스템에 전달하는 모든 신호.

• 콜백 함수: 특정 이벤트가 발생했을 때 시스템에 의해 자동으로 호출되는 함수입니다. 예를 들어, cv.setMouseCallback()은 사용자가 마우스를 조작할 때마다 우리가 정의한 paint나 draw_roi 함수를 실행.

• 루프(Loop)와 입력 감지: 실시간으로 변화를 반영하기 위해 while True 루프를 사용하며, 루프 안에서 cv.waitKey() 함수를 통해 사용자가 어떤 키를 눌렀는지 0.001초 단위로 감시.

### 2. 전체 코드

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
### 3. 핵심 코드

• cv.setMouseCallback('Paint', paint): 특정 창에서 발생하는 마우스 동작(클릭, 이동 등)을 감지하여 미리 정의한 함수가 실행되도록 연결.

• cv.circle(img, (x, y), brush_size, color, -1): 마우스 좌표 (x, y)를 중심으로 원을 그립니다. 마지막 인자가 -1이면 선이 아닌 속이 꽉 찬 원을 그리게 되어 붓질 효과를 냄.

• cv.waitKey(1) & 0xFF: 아주 짧은 시간(1ms) 동안 키보드 입력을 대기합니다. 루프 안에서 사용되어 실시간으로 +, -, q 등의 키 입력을 구분하고 반응하게 함.

### 4.실행 결과

![02result](https://github.com/user-attachments/assets/b9b02936-db1e-47d8-8217-cdba799c95d4)


---
## 🌀 문제 03 마우스로 영역 선택 및 ROI(관심영역) 추출

### 1. 주요 개념

![33](https://github.com/user-attachments/assets/cf1cc68e-6a51-4597-99e0-5a9a105a942f)

전체 이미지 중에서 우리가 집중하고 싶은 특정 부분만을 골라내는 기술.

• ROI (Region of Interest): 영상 전체가 아닌 분석이나 처리가 필요한 특정 부분 영역을 의미.

• 좌표 시스템: 이미지의 왼쪽 상단 끝이 (0, 0)이며, 오른쪽으로 갈수록 x값이 커지고 아래로 갈수록 y값이 커짐.

• Numpy 슬라이싱: 이미지는 행렬 데이터이므로, img[시작_y:끝_y, 시작_x:끝_x] 형식을 사용하여 원하는 범위의 데이터만 잘라낼 수 있습니다. 이것은 마치 거대한 표에서 특정 행과 열만 가위로 오려내는 것과 같은 원리.

### 2. 전체 코드

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
### 3.핵심 코드

• Numpy 슬라이싱 (img[y1:y2, x1:x2]): 사진 데이터 행렬에서 특정 행(y)과 열(x) 범위만 잘라냅니다. 이것이 OpenCV에서 관심 영역(ROI)을 추출하는 표준 방법.

• cv.imshow('ROI', roi): 메모리에 있는 이미지 행렬을 이미지 파일 형식으로 저장.

• img.copy(): 원본 이미지를 복제합니다.  드래그할 때마다 사각형 잔상이 남지 않도록 깨끗한 원본을 매번 다시 불러와 그리는 기법이 중요.

### 4. 실행 결과

![03result](https://github.com/user-attachments/assets/d2b9abc4-c31e-4c1a-adea-146437894322)

---
