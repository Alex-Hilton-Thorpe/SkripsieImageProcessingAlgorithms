# overlay_tracker.py

import cv2

# Video and output paths
video_path = "Tennis_Yellow_Ball_v2.mp4"
output_coords_path = "outputs.txt"
output_video_path = "tracked_video.mp4"

# Load coordinates from outputs.txt
coords = []
with open(output_coords_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            coords.append((-1, -1))  # no detection
            continue
        parts = line.split(",")
        if len(parts) != 2:
            coords.append((-1, -1))
            continue
        try:
            x = float(parts[0]) * 1920  # scale x
            y = (float(parts[1])) * 1080  # scale y
            coords.append((int(x), int(y)))
        except ValueError:
            coords.append((-1, -1))

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(coords):
        x, y = coords[frame_idx]
        if x != -1 and y != -1:
            # Draw small tracker circle
            cv2.circle(frame, (x, y), radius=8, color=(0, 0, 255), thickness=-1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tracked video saved to {output_video_path}")
