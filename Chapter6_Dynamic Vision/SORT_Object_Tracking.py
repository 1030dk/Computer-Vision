import sys
import cv2
import numpy as np
from sort import Sort # 같은 폴더에 있는 sort.py 파일에서 가져옵니다.

# 1. YOLOv3 모델 로드 
try:
    # 사전에 학습된 YOLOv3 가중치와 설정 파일을 불러옵니다.
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except cv2.error:
    print("🚨 YOLO 가중치 또는 설정 파일(yolov3.weights, yolov3.cfg)을 찾을 수 없습니다.")
    sys.exit()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 2. SORT 추적기 초기화 [cite: 19]
# max_age: 객체가 화면에서 5프레임 동안 가려져도 기존 ID를 유지합니다.
# min_hits: 최소 3번 연속으로 검출되어야 실제 객체로 인정하고 추적을 시작합니다.
mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3) 

# 3. 실습 비디오 파일 열기 
video_path = "slow_traffic_small.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"🚨 비디오 파일을 열 수 없습니다: {video_path}")
    sys.exit()

print("✅ 다중 객체 추적을 시작합니다. (종료: ESC 키)")

# 비디오 프레임 반복 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # YOLO 입력을 위한 이미지 정규화 및 변환 (Blob) [cite: 18, 24]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []

    # 검출 결과 분석
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 신뢰도가 50%(0.5) 이상인 객체(사람, 자동차 등)만 추출 [cite: 18]
            if confidence > 0.5: 
                # YOLO의 중심점 좌표를 픽셀 단위로 변환
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 좌상단 좌표(x, y) 계산
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # 노이즈 및 중복 박스 제거 (NMS 알고리즘)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # SORT 알고리즘 입력 형식에 맞게 데이터 재구성: [[좌상단x, 좌상단y, 우하단x, 우하단y, 신뢰도], ...]
    detections = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            detections.append([x, y, x + w, y + h, confidences[i]])

    # 데이터 형식 맞추기 (검출된 객체가 없으면 빈 배열 생성)
    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    # 4. 객체 추적 및 ID 갱신 (SORT 연관 수행) [cite: 20]
    track_bbs_ids = mot_tracker.update(detections)

    # 5. 결과 시각화 (ID 및 경계 상자 표시) [cite: 21]
    for track in track_bbs_ids:
        x1, y1, x2, y2, obj_id = [int(v) for v in track]
        
        color = (0, 255, 0) # 연두색
        
        # 객체 테두리(경계 상자) 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # ID 텍스트 및 배경 박스 그리기 (가독성 향상)
        label = f"ID: {obj_id}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), color, -1) # 텍스트 배경
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # 검은색 텍스트

    # 완성된 프레임을 화면에 출력 [cite: 21]
    cv2.imshow("SORT Object Tracking", frame)

    # ESC 키를 누르면 종료 (27 = ESC 아스키 코드)
    if cv2.waitKey(1) == 27:
        break

# 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()

# Mac 환경에서 OpenCV 창이 정상적으로 닫히지 않는 버그 방지용
for i in range(1, 5):
    cv2.waitKey(1)