#Algorithm 4: MOG2 Background Subtraction + Tuned Yellow Mask Tracker with Normalized Output
import cv2
import numpy as np

# --- Parameters ---
video_path = "Tennis_Yellow_Ball_v2.mp4"
output_file_path = "alg4.txt"
min_ball_area = 10
max_ball_area = 250

# Tuned HSV range for yellow tennis ball (avoid skin)
lower_yellow = np.array([25, 120, 150])
upper_yellow = np.array([40, 255, 255])

# --- Load video ---
vid = cv2.VideoCapture(video_path)
ret, frame = vid.read()
if not ret:
    print("Error: Could not read video")
    exit()

fps = vid.get(cv2.CAP_PROP_FPS)
display_delay = int(1000 / fps)

# --- Background subtractor ---
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

previous_ball_position = None

# --- Kernel for morphological operations ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# --- Open output file ---
output_file = open(output_file_path, "w")

# --- Tracking loop ---
while True:
    ret, frame = vid.read()
    if not ret:
        break

    # 1. Background subtraction
    fg_mask = back_sub.apply(frame)

    # 2. Apply tuned yellow mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Optional: show yellow mask for debugging
    cv2.imshow("Yellow Mask Only", color_mask)

    # 3. Combine masks
    combined_mask = cv2.bitwise_and(fg_mask, color_mask)

    # 4. Remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Optional: show combined mask
    cv2.imshow("Yellow + MOG2 Mask", combined_mask)

    # 5. Find contours (potential balls)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_ball_area <= area <= max_ball_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ball_candidates.append((cx, cy, area))

    # 6. Select candidate closest to previous position
    selected_ball = None
    if previous_ball_position and ball_candidates:
        prev_x, prev_y = previous_ball_position
        min_dist = float('inf')
        for cx, cy, _ in ball_candidates:
            dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
            if dist < min_dist:
                min_dist = dist
                selected_ball = (cx, cy)
    elif ball_candidates:
        selected_ball = (ball_candidates[0][0], ball_candidates[0][1])

    # 7. Draw detected ball on original frame
    if selected_ball:
        cx, cy = selected_ball
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        previous_ball_position = selected_ball

        # 8. Write normalized coordinates to file
        output_file.write(f"{cx/1920},{cy/1080}\n")
    else:
        previous_ball_position = None
        # No detection, write -1,-1
        output_file.write("-1,-1\n")

    # 9. Show the tracking frame
    cv2.imshow("MOG2 + Tuned Yellow Ball Tracker", frame)
    if cv2.waitKey(display_delay) & 0xFF == ord('q'):
        break

output_file.close()
vid.release()
cv2.destroyAllWindows()
