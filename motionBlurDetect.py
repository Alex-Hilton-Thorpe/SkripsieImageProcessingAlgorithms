# Algorithm 2: Blur + Yellow Hybrid Tracker
import cv2
import numpy as np

# --- Parameters ---
video_path = "Tennis_Yellow_Ball_v2.mp4"
output_file_path = "alg2.txt"

threshold_diff = 15   # motion difference threshold
min_area = 5          # minimum contour area
max_area = 150        # maximum area to ignore players
aspect_ratio_min = 0.5
aspect_ratio_max = 2.0
min_speed = 2         # minimum speed in pixels/frame
display_delay = 50    # ms

# Yellow color range in HSV
lower_yellow = np.array([20, 100, 150])
upper_yellow = np.array([35, 255, 255])

# Open video
vid = cv2.VideoCapture(video_path)
if not vid.isOpened():
    print("Error: Could not open video file")
    exit()

output_file = open(output_file_path, "w")

# Previous grayscale frame and centroid
ret, prev_frame = vid.read()
if not ret:
    print("Error: Could not read first frame")
    exit()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_cx, prev_cy = None, None
frame_count = 1

while True:
    ret, frame = vid.read()
    if not ret:
        break
    frame_count += 1

    # --- Motion mask ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, threshold_diff, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel)

    # --- Yellow mask ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # --- Combine masks ---
    combined_mask = cv2.bitwise_and(motion_mask, yellow_mask)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if aspect_ratio_min <= aspect_ratio <= aspect_ratio_max:
                cx = int(x + w/2)
                cy = int(y + h/2)

                # Speed filter
                if prev_cx is not None and prev_cy is not None:
                    speed = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    if speed < min_speed:
                        continue

                output_file.write(f"{cx/1920},{cy/1080}\n")
                ball_detected = True

                # Draw bounding box and centroid
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                prev_cx, prev_cy = cx, cy
                break

    if not ball_detected:
        output_file.write("-1,-1\n")
        prev_cx, prev_cy = None, None

    # Show tracking
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(display_delay) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()

output_file.close()
vid.release()
cv2.destroyAllWindows()
print(f"Tracking complete. Output written to {output_file_path}")
