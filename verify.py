import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import cv2
from preprocess import img_preprocess

model = tf.keras.models.load_model('/Users/sudipkhadka/Desktop/Face-Recognition/siameseModel.keras')

# Build Verify Function

def verify_image(model):
    results = []
    for image in os.listdir(os.path.join('crediential_val_data', 'verify_image')):
        input_img = img_preprocess(os.path.join('crediential_val_data', 'input_image', 'input_image.jpg'))
        validation_img = img_preprocess(os.path.join('crediential_val_data', 'verify_image', image))
        
 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

        detection_threshold = 0.80
        verification_threshold = 0.65
   
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('crediential_val_data', 'verify_image'))) 
    verified = verification > verification_threshold
    
    return results, verified


def capture_and_verify():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[150:575, 500:880, :]
    
        cv2.imshow('Verifying.....', frame)
    
        # Verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):
            # Save input image to application_data/input_image folder 
            cv2.imwrite(os.path.join('crediential_val_data', 'input_image', 'input_image.jpg'), frame)
            # Run verification
            results, verified = verify_image(model)
            print(verified)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break  
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_verify()
