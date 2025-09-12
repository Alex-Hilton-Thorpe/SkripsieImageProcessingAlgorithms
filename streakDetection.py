# Algorithm 6: Frame-Local Streak Tracker (No Previous Position)
import cv2
import numpy as np

# ---------------- Parameters ----------------
video_path = "Tennis_Yellow_Ball_v2.mp4"

# HSV yellow range (tune if needed)
lower_yellow = np.array([20, 80, 100])
upper_yellow = np.array([40, 255, 255])

# Morphology kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Blob / streak thresholds
min_area = 5
max_area = 1200       # ignore large objects (players)
min_streak_length = 4 # pixels

frame_width  = 1920
frame_height = 1080

output_file = open("alg6.txt", "w")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Yellow mask ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)

    # --- Find blobs ---
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        length = max(w, h)
        short  = min(w, h)
        aspect_ratio = length / (short + 1e-5)

        # Compute centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])

        # Score: prefer elongated if long, otherwise prefer compact
        score = aspect_ratio * length if aspect_ratio > 1.3 else area
        candidates.append((cx, cy, score))

    # --- Pick best candidate ---
    if candidates:
        selected = max(candidates, key=lambda c: c[2])
        cx, cy, _ = selected

        cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)

        norm_x = cx / frame_width
        norm_y = cy / frame_height
        output_file.write(f"{norm_x:.6f} {norm_y:.6f}\n")
    else:
        output_file.write("-1 -1\n")

    # --- Show for debugging ---
    cv2.imshow("Yellow Mask", yellow_mask)
    cv2.imshow("Tracked Frame", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

output_file.close()
cap.release()
cv2.destroyAllWindows()
