import random
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import uuid
import time

def create_user_folders(base_path):
    """Create unique folders for a new user."""
    user_id = str(uuid.uuid4())[:8]  # Generate a unique user identifier
    user_anchor_path = os.path.join(base_path, 'anchor', user_id)
    user_positive_path = os.path.join(base_path, 'positive', user_id)
    user_negative_path = os.path.join(base_path, 'negative', user_id)

    os.makedirs(user_anchor_path, exist_ok=True)
    os.makedirs(user_positive_path, exist_ok=True)
    os.makedirs(user_negative_path, exist_ok=True)

    return user_anchor_path, user_positive_path

def capture_new_user():
    base_path = 'data'
    anchor_path, positive_path = create_user_folders(base_path)

    center = (695, 330)  
    axes = (290, 290)    
    angle = 0            

    dash_angle_step = 15  
    dash_length_angle = 12  

    cam_pic = cv2.VideoCapture(0)

    if not cam_pic.isOpened():
        print("Error: Camera could not be opened.")
        return

    start_time = time.time()
    images_taken = 0
    max_images = 150
    capture_duration = 5

    while cam_pic.isOpened():
        ret, frame = cam_pic.read()

        if not ret:
            break

        for theta in range(0, 360, dash_angle_step + dash_length_angle):
            start_angle = theta
            end_angle = theta + dash_length_angle
            cv2.ellipse(frame, center, axes, angle, start_angle, end_angle, (0, 255, 0), 2)

        elapsed_time = time.time() - start_time
        if elapsed_time <= capture_duration and images_taken < max_images:
            # Add countdown timer
            timer_text = f"Time left: {max(0, capture_duration - int(elapsed_time))}s"
            cv2.putText(frame, timer_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save frame
            imagename = os.path.join(anchor_path, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imagename, frame[150:575, 500:880, :])
            images_taken += 1
        else:
            break  # Exit the loop after 5 seconds or 200 images

        cv2.imshow("Scanning Your Face....", frame)

        key = cv2.waitKey(1) & 0xFF  
        if key == ord('q'):
            break

    print(f"Captured {images_taken} images.")
    cam_pic.release()
    cv2.destroyAllWindows()

# Call the function to start capturing
capture_new_user()
