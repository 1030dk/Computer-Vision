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