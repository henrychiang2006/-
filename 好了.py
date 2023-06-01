import cv2
import mediapipe as mp
import math

def calculate_angle(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def detect_hand_angle():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoint_pos = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    keypoint_pos.append((x, y))
                
                if len(keypoint_pos) >= 13:
                    # 計算食指和中指之間的向量
                    v_index = (keypoint_pos[8][0] - keypoint_pos[6][0], keypoint_pos[8][1] - keypoint_pos[6][1])
                    v_middle = (keypoint_pos[12][0] - keypoint_pos[10][0], keypoint_pos[12][1] - keypoint_pos[10][1])
                    
                    # 計算食指和中指之間的角度
                    angle = calculate_angle(v_index, v_middle)
                    
                    # 顯示食指和中指之間的角度
                    cv2.putText(frame, f"Index-Middle Angle: {angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Angle Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_hand_angle()
