## 🌀 문제 01 SORT 알고리즘을 활용한 다중 객체 추적기 구현


<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/5f8ad49c-5ff5-4e14-9f86-4bc254d4d2de" />



### 1.주요 개념

- **객체 검출 (Detection - YOLOv3)**: 영상은 수많은 정지 사진(프레임)의 연속입니다. YOLO는 영상의 흐름을 모르고 오직 현재 주어진 사진 1장만 보고 사물(사람, 자동차 등)의 위치(경계 상자)를 찾아냄.
- **노이즈 제거 (NMS - Non-Maximum Suppression): "정제" 역할**: YOLO는 가끔 자동차 1대에 여러 개의 박스를 겹쳐서 그리는 실수를 하는데 NMS는 겹쳐진 여러 박스 중 가장 정확도(신뢰도)가 높은 1개만 남기고 나머지를 지워버림.
- **위치 예측 (Kalman Filter)**: SORT 알고리즘의 첫 번째 핵심입니다. 이전 프레임에서 자동차가 움직이던 속도와 방향을 기억해 두었다가 미리 예측함.
- **데이터 연관 (Hungarian Algorithm & IoU)**: 만 필터가 '예측한 위치'와 이번 프레임에서 YOLO가 새로 찾은 '실제 위치'가 얼마나 겹치는지(IoU)를 계산합니다. 가장 많이 겹치는 쌍을 찾아 동일한 고유 ID를 부여함.




### 2. 전체 코드

```Python
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
```


### 3. 핵심 코드
- **`if confidence > 0.5: 
    boxes.append([x, y, w, h])`** : AI가 찾은 사물 중 "내가 50% 이상 확신하는 것"만 리스트에 넣음.
  
- **`indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)`** : 물체에 겹쳐진 여러 개의 네모 박스 중, 가장 정확한 1개만 남기고 나머지를 지움.
  
- **`track_bbs_ids = mot_tracker.update(detections)`** : 현재 프레임의 위치 데이터(detections)를 넣어주면, 과거의 데이터와 비교하여 같은 객체임을 증명하는 고유 ID를 붙여서 반환함.








### 4. 실행 결과


https://github.com/user-attachments/assets/ecf4925a-6e1e-437d-95e3-ce1e676a6337


---

## 🌀 문제 02 Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화


<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/c34b9834-1a81-4429-9412-5ef8b98d3c0d" />



### 1. 주요 개념

- **얼굴 메시 (Face Mesh)**: 단순하게 얼굴을 네모 박스로 찾는 것을 넘어, 눈, 코, 입, 턱선 등 얼굴의 입체적인 표면을 덮는 **468개의 정밀한 거미줄(Mesh)을 생성합니다. 이를 통해 눈 깜박임이나 입 모양의 변화까지 실시간으로 추적함
- **파이프라인 (Detection + Landmark)**: 구글의 Mediapipe는 성능을 극대화하기 위해 내부적으로 두 번의 계산함.
  - 1차: 아주 가벼운 AI가 화면에서 얼굴이 있는 대략적인 '위치(박스)'만 빠르게 찾음
  - 2차: 찾아낸 얼굴 영역 안에서만 정밀한 AI가 468개의 특징점(Landmark)을 찍어냄. (얼굴을 한 번 찾고 나면, 다음 프레임부터는 1차 과정을 생략하고 점만 따라가므로 빠르다.)   
- **정규화된 좌표계 (Normalized Coordinates)**: Mediapipe AI는 화면 전체 가로 길이의 0.6(60%) 위치에 있다라고 비율(0.0 ~ 1.0)로 알려주기에 영상 크기가 4K, 스마트폰 크기와 상관없이 똑같이 정확하게 작동함.


### 2. 전체 코드

```Python
import cv2
import mediapipe as mp 

# 1. Mediapipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,               # 화면에서 찾을 최대 얼굴 개수 (1명)
    refine_landmarks=True,         # 눈동자/입술 주변 정밀 추적 활성화
    min_detection_confidence=0.5,  # 초기 얼굴 검출 신뢰도 (50% 이상)
    min_tracking_confidence=0.5    # 추적 유지 신뢰도
)

# 2. 웹캠 캡처 시작 (1번은 Mac 외부/추가 카메라, 안 되면 0번으로 변경)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("🚨 카메라를 열 수 없습니다. 시스템 설정에서 카메라 권한을 확인하세요.")

print("✅ 실행 중... 카메라를 바라보세요! (ESC를 누르면 종료됩니다)")

# 3. 실시간 프레임 반복 처리
while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:             
        break

    # 카메라 화면 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)

    # Mediapipe 처리를 위해 BGR(OpenCV 기본)을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 얼굴 468개 랜드마크 추출 수행
    results = face_mesh.process(rgb_frame)

    # 4. 결과 시각화 (얼굴이 인식되었을 때만 실행)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape # 현재 화면의 세로, 가로 크기
            
            # 468개의 점을 하나씩 화면에 그리기
            for landmark in face_landmarks.landmark:
                # 0~1 사이의 비율로 나온 좌표를 실제 픽셀 위치로 변환
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 얼굴에 연두색(0, 255, 0) 점 찍기
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

    # 완성된 화면 출력
    cv2.imshow('Face Landmark', frame)
    
    # 5. 종료 조건 (ESC 키)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

# 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()

# Mac 환경에서 창이 정상적으로 닫히도록 딜레이 추가
for i in range(1, 5):
    cv2.waitKey(1)
```


### 3. 핵심 코드

- **`face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,        
    refine_landmarks=True,        
    min_detection_confidence=0.5
)`** : 수많은 사람의 얼굴을 다 찾지않고, max_num_faces=1로 설정하여 실습자의 얼굴 1개에만 연산력을 집중하게 만드는 설정.
- **`frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`** : OpenCV 라이브러리는 파란색(B)부터 읽는 특이한 습관(BGR)이 있지만, 구글의 Mediapipe AI는 빨간색(R)부터 읽는 표준(RGB)으로 학습되어서, 이 변환 코드가 없으면 AI가 얼굴을 찾지 못 할 수 있다.
- **`results = face_mesh.process(rgb_frame)`** :딥러닝 모델이 작동하여 눈썹의 휘어짐, 입꼬리의 위치 등을 3차원 데이터로 완벽하게 계산함.

- **`cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)`** :  방금 계산한 (cx, cy) 좌표에 OpenCV의 원 그리기 함수(circle)를 써서 점을 찍어줌.


### 4. 실행 결과


![result2](https://github.com/user-attachments/assets/6d44e1a6-40d1-4dd9-ba0e-91ca8e97c4c1)


---

