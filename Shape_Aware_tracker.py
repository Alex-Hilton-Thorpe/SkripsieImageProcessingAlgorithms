# Algorithm 5: Motion + Yellow + Shape-Aware Tracker with First-Throw Fix
import cv2
import numpy as np

# ---------------- Parameters ----------------
lower_yellow = np.array([20, 100, 150])
upper_yellow = np.array([35, 255, 255])

# Extended range for streaks / blurred ball
lower_yellow_ext = np.array([15, 80, 120])
upper_yellow_ext = np.array([40, 255, 255])

min_area = 5
max_area = 150        # max_area reduced to avoid player
aspect_ratio_threshold = 1.5

throwup_frames = 20  # number of frames to use relaxed detection for first throw

frame_width  = 1920
frame_height = 1080

# ---------------- State ----------------
prev_gray = None
prev_position = None
frame_count = 0

# ---------------- Input / Output ----------------
video_path = "Tennis_Yellow_Ball_v2.mp4"
output_path = "alg5.txt"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

out_file = open(output_path, "w")

# ---------------- Main loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- First throwup: relax motion / area restrictions ---
    if frame_count <= throwup_frames:
        combined_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_yellow, upper_yellow),
                                       cv2.inRange(hsv, lower_yellow_ext, upper_yellow_ext))
        temp_min_area = 2  # smaller min area to catch small ball
    else:
        # --- Motion mask ---
        if prev_gray is None:
            prev_gray = gray
            out_file.write("-1,-1\n")
            cv2.imshow("Alg5 Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        # --- Combine motion + color ---
        color_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_yellow, upper_yellow),
                                    cv2.inRange(hsv, lower_yellow_ext, upper_yellow_ext))
        combined_mask = cv2.bitwise_and(motion_mask, color_mask)
        temp_min_area = min_area

    # --- Find blobs ---
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < temp_min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w/h, h/w)

        # --- Streak / elongated blob ---
        if aspect_ratio >= aspect_ratio_threshold or area > 100:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (ex, ey), (ew, eh), angle = ellipse
                if ew > 0 and eh > 0:
                    cx, cy = int(ex), int(ey)
                    cv2.ellipse(frame, ellipse, (0,255,255), 2)
                else:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                    else:
                        continue
            else:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                else:
                    continue
        else:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                continue

        candidates.append((cx, cy, area))

    # --- Select closest candidate to previous ball ---
    if candidates:
        if prev_position is None:
            selected = max(candidates, key=lambda c: c[2])
        else:
            selected = min(candidates, key=lambda c: np.linalg.norm(np.array(c[:2])-np.array(prev_position)))

        cx, cy, _ = selected
        prev_position = (cx, cy)

        # Draw green circle at ball
        cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

        # Normalize output
        norm_x = cx / frame_width
        norm_y = cy / frame_height
        out_file.write(f"{norm_x:.6f},{norm_y:.6f}\n")
    else:
        prev_position = None
        out_file.write("-1,-1\n")

    prev_gray = gray.copy()

    # --- Display ---
    cv2.imshow("Alg5 Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- Cleanup ----------------
out_file.close()
cap.release()
cv2.destroyAllWindows()
