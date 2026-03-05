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