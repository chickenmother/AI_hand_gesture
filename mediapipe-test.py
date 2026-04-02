import cv2
import numpy as np
import mediapipe as mp

# MediaPipe (顔検出)と手の検出を準備
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 顔検出と手検出のインスタンスを作成
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
hand_detection = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Webカメラの映像をキャプチャ
cap = cv2.VideoCapture(0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 画像データをMediaPipeで処理するためにRGB形式に変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # 顔検出処理
    face_results = face_detection.process(image)
    
    # 手検出処理
    hand_results = hand_detection.process(image)
    
    # 描画用にRGBからBGRに再変換
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 顔の位置を描画
    if face_results.detections:
        for detection in face_results.detections:
            face_points = detection.location_data.relative_keypoints
            for i in range(len(face_points)):
                x = int(face_points[i].x * image.shape[1])
                y = int(face_points[i].y * image.shape[0])
                
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # 半径5の黄色の点を描画
    
    # 手の位置を描画
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
    
    # 映像を表示
    cv2.imshow('Camera Stream', image)
    
    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
