import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

class VideoClassifier:
    def __init__(self, model_path, class_names, num_frames=8, resolution=224, model_id='a0', num_classes=10):
        self.num_frames = num_frames
        self.resolution = resolution
        self.frame_buffer = deque(maxlen=num_frames)
        self.class_names = class_names
        
        # Build MoViNet model
        backbone = movinet.Movinet(model_id=model_id)
        backbone.trainable = False
        self.model = self.build_classifier(backbone, num_classes)
        self.model.build([1, num_frames, resolution, resolution, 3])

        # Load model weights
        self.model.load_weights(model_path)

    def build_classifier(self, backbone, num_classes):
        model = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=num_classes
        )
        return model

    def format_frames(self, frame, output_size):
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, *output_size)
        return frame

    @tf.function
    def predict(self, input_frames):
        return self.model(input_frames, training=False)

    def classify_frame(self, frame):
        # Preprocess the frame and add to the buffer
        processed_frame = self.format_frames(frame, (self.resolution, self.resolution))
        self.frame_buffer.append(processed_frame)

        if len(self.frame_buffer) == self.num_frames:
            # Prepare input for the model
            model_input = tf.stack(list(self.frame_buffer))
            model_input = tf.expand_dims(model_input, axis=0)

            # Perform the classification
            prediction = self.predict(model_input)
            predicted_class = self.class_names[np.argmax(prediction[0])]
            confidence = tf.nn.softmax(prediction[0])[np.argmax(prediction[0])].numpy()

            return predicted_class, confidence

        return None, None





import cv2
import numpy as np
import sys
import threading
from ultralytics import YOLO
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot
from MetalTheft.motion_detection import detect_motion
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig

# Global variables
motion_detected_flag = False
start_time = None
counter = 1

# Initialize Classes
email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
aws = AWSConfig()
model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.pt')  
rtsp_url = RTSP_URL
cap = cv2.VideoCapture(rtsp_url)

# Initialize VideoClassifier with appropriate parameters
class_names = ["CricketBowling", "JavelinThrow", "ThrowDiscuss"]
video_classifier = VideoClassifier(model_path="MetalTheft/Movinet_Model.keras", class_names=class_names)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    sys.exit()

cv2.namedWindow('IP Camera Feed')
cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point, param=None)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to receive frame from RTSP stream. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if roi_selector.is_roi_selected():
            roi_pts_np = roi_selector.get_roi_points()

            combined_frame, thresh, person_detected,_ = detect_motion(
                frame, blurred_frame, model, fgbg, roi_pts_np)

            motion_in_roi = cv2.countNonZero(thresh) > 400

            if motion_in_roi:  # or person_detected
                if not motion_detected_flag:
                    snapshot_path = save_snapshot(combined_frame)
                    if snapshot_path:
                        threading.Thread(target=email.send_alert_email, args=(snapshot_path,)).start()
                        current_time = datetime.now()
                        # snapshot_url = aws.upload_to_s3_bucket(snapshot_path)
                        # mongo_handler.save_to_mongo(snapshot_url, current_time)

                    start_time = current_time
                    motion_detected_flag = True

            else:
                if motion_detected_flag:
                    end_time = datetime.now()
                    if (end_time - start_time).total_seconds() > 1:
                        counter += 1
                    motion_detected_flag = False

            # Perform video classification
            predicted_class, confidence = video_classifier.classify_frame(frame)
            if predicted_class:
                cv2.putText(combined_frame, f"{predicted_class}: {confidence:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            roi_color = (0, 0, 255) if motion_detected_flag else (0, 255, 0)

            motion_mask = np.zeros(combined_frame.shape, dtype=np.uint8)
            cv2.fillPoly(motion_mask, [roi_pts_np], roi_color)

            alpha = 0.8
            combined_frame = cv2.addWeighted(combined_frame, alpha, motion_mask, 1 - alpha, 0)

            cv2.imshow('Motion', thresh)
            cv2.imshow("imgRegion", combined_frame)

        else:
            cv2.imshow('IP Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # 'r' key to reset the ROI
            roi_selector.reset_roi()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise MetalTheptException(e, sys) from e

cap.release()
cv2.destroyAllWindows()


