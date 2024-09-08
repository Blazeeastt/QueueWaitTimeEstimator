import cv2
import numpy as np
from collections import deque
import time

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture for Mac webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Queue parameters
queue_history = deque(maxlen=30)  # Store recent queue positions
min_queue_length = 1  # Reduced minimum number of people to qconsider it a queue

avg_service_time = 1  # Average time to serve one person (in minutes)

update_interval = 2  # Update wait time and queue length every 2 seconds
last_update_time = time.time()

def detect_queue(faces):
    """Detect the queue based on face positions"""
    if len(faces) < min_queue_length:
        return None
    
    # Sort faces by y-coordinate (assuming queue is roughly vertical)
    sorted_faces = sorted(faces, key=lambda f: f[1])
    
    # Calculate the average x-coordinate of the queue
    avg_x = sum(f[0] for f in sorted_faces) / len(sorted_faces)
    
    # Define queue boundaries with more tolerance
    queue_top = max(0, sorted_faces[0][1] - 50)
    queue_bottom = min(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), sorted_faces[-1][1] + sorted_faces[-1][3] + 50)
    queue_left = max(0, int(avg_x - 150))
    queue_right = min(cap.get(cv2.CAP_PROP_FRAME_WIDTH), int(avg_x + 150))
    
    return (queue_left, queue_top, queue_right, queue_bottom)

def estimate_wait_time(num_people):
    """Estimate total wait time based on number of people"""
    return num_people * avg_service_time

def draw_text(img, text, position, font_scale=0.8, thickness=2):
    """Draw text with a dark background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# Initialize variables for periodic updates
current_queue_count = 0
current_wait_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Adjusted parameters for more lenient face detection

    queue_area = detect_queue(faces)
    if queue_area:
        queue_history.append(queue_area)
    
    if queue_history:
        # Use average of recent queue positions for stability
        avg_queue = np.mean(queue_history, axis=0).astype(int)
        cv2.rectangle(frame, (avg_queue[0], avg_queue[1]), (avg_queue[2], avg_queue[3]), (0, 255, 255), 2)

        # Update queue count and wait time every 2 seconds
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            current_queue_count = sum(1 for (x, y, w, h) in faces if avg_queue[0] <= x <= avg_queue[2] and avg_queue[1] <= y <= avg_queue[3])
            current_wait_time = estimate_wait_time(current_queue_count)
            last_update_time = current_time

        draw_text(frame, f'People in queue: {current_queue_count}', (10, 30))
        draw_text(frame, f'Estimated wait time: {current_wait_time} minutes', (10, 70))
    else:
        draw_text(frame, 'No queue detected', (10, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Webcam Queue Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()