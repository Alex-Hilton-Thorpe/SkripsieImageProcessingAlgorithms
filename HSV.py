#Algorithm 1: HSV image procesing algorithm
import cv2
import numpy as np

video_path = "Tennis_Yellow_Ball_v2.mp4"
vid = cv2.VideoCapture(video_path)

lower_yellow = np.array([20, 100, 150])
upper_yellow = np.array([35, 255, 255])

# Open output file for writing
output_file = open("alg1.txt", "w")

while True:
    ret, frame = vid.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 5]  # ignore tiny noise

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Draw rectangle (optional visualization)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Write x, y of the center of the bounding box to file
        cx = x + w // 2
        cy = y + h // 2
        output_file.write(f"{cx/1920},{cy/1080}\n")
    else:
        # No detection
        output_file.write("-1,-1\n")

    cv2.imshow("Yellow Ball Tracker", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Close resources
vid.release()
cv2.destroyAllWindows()
output_file.close()
