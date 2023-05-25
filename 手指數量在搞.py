import cv2
import numpy as np

def segment_hand(frame):
    # 將影像轉換為HSV色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定義膚色的範圍（可根據需要進行調整）
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # 根據膚色範圍創建遮罩
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 執行形態學操作，以去除噪音
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 應用遮罩到原始影像上，僅保留手部區域
    segmented = cv2.bitwise_and(frame, frame, mask=mask)

    return segmented

def detect_finger_spacing(hand_contour):
    # 找到手部輪廓的凸包
    hull = cv2.convexHull(hand_contour, returnPoints=False)

    # 找到凸包的凸缺陷
    defects = cv2.convexityDefects(hand_contour, hull)

    # 計算手指數量
    finger_count = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])

            # 計算三角形的邊長
            a = np.linalg.norm(np.array(start) - np.array(far))
            b = np.linalg.norm(np.array(end) - np.array(far))
            c = np.linalg.norm(np.array(start) - np.array(end))

            # 使用三角形的餘弦定理計算夾角
            angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

            # 如果夾角小於指定閾值，則將其視為一根手指
            if angle < np.pi / 2:
                finger_count += 1

    return finger_count

# 擷取影像並處理手部偵測及手指間距
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # 分割手部
    segmented = segment_hand(frame)

    # 找到手部輪廓
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的輪廓（手部）
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)

        # 偵測手指間距
        finger_count = detect_finger_spacing(hand_contour)

        # 在影像上顯示手指數量
        cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 顯示原始影像
    cv2.imshow("Hand Finger Spacing", frame)

    # 按下 'q' 鍵結束迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉視窗
cap.release()
cv2.destroyAllWindows()
