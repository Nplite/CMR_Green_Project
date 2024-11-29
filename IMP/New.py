import cv2
import numpy as np
# import winsound
import threading
import pandas as pd
import datetime
import smtplib
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the Excel file
df = pd.read_excel('Motion_detection.xlsx', sheet_name='Sheet1')

motion_detected_flag = False
start_time = None
counter = 1

# Initialize video capture
cap = cv2.VideoCapture('DATA/npl.mp4')
# mask = cv2.imread('./Videos/mask.png', 0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
roi_pts = [517, 116, 529, 124, 552, 713, 384, 719]
#roi_pts = [512, 123, 392, 715, 682, 718, 611, 123]

def draw_boxes(frame, detections):
    try:
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 0:  
                    conf = float(conf)  
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    #cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame
    except Exception as e:
        print(f"Error in draw_boxes: {e}")




while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Create the ROI mask
        roi_pts_np = np.array(roi_pts, np.int32)
        roi_pts_np = roi_pts_np.reshape((-1, 1, 2))
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_np], (255, 255, 255))

        # Combine the original frame and the blurred frame using the ROI mask
        mask_inv = cv2.bitwise_not(mask)
        frame_roi = cv2.bitwise_and(frame, mask_inv)
        blurred_roi = cv2.bitwise_and(blurred_frame, mask)
        combined_frame = cv2.add(frame_roi, blurred_roi)
        
        # Use YOLOv8 for object detection
        results = model.predict(combined_frame)

        annotator = Annotator(combined_frame)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = box.cls
                if c == 0:
                    b = box.xyxy[0]  
                    detections.append((b[0], b[1], b[2], b[3], float(box.conf), c))
                    annotator.box_label(b, model.names[int(c)])

        combined_frame = annotator.result()


        # Check if any persons are detected
        person_detected = any(box.cls == 0 for r in results for box in r.boxes)

        if motion_detected_flag:
            roi_color = (0, 0, 255)  # Red
        else:
            roi_color = (0, 255, 0)  # Green

        # Create a mask image filled with the appropriate color
        motion_mask = np.zeros(combined_frame.shape, dtype=np.uint8)
        cv2.fillPoly(motion_mask, [roi_pts_np], roi_color)

        # Apply the mask to the frame using cv2.addWeighted()
        alpha = 0.8
        combined_frame = cv2.addWeighted(combined_frame, alpha, motion_mask, 1 - alpha, 0)

        # Apply background subtraction to the polyline area
        fgmask = fgbg.apply(combined_frame)
        _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
        mask = np.zeros(fgmask.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_np], (255, 255, 255))
        thresh = cv2.bitwise_and(thresh, mask)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Check if there is motion in the ROI
        motion_in_roi = cv2.countNonZero(thresh) > 300

        if motion_in_roi and person_detected:
            if not motion_detected_flag:
                # threading.Thread(target=play_alarm).start()
                #send_sms()
                start_time = datetime.datetime.now()
                video_name = 'motion_' + str(counter) + '.mp4'
                out = cv2.VideoWriter(video_name, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                motion_detected_flag = True
        else:
            if motion_detected_flag:
                end_time = datetime.datetime.now()
                if (end_time - start_time).total_seconds() > 1:
                    df = pd.concat([df, pd.DataFrame({'Start Time': [start_time.strftime('%d-%m-%Y %H:%M:%S')],
                                                    'End Time': [end_time.strftime('%d-%m-%Y %H:%M:%S')],
                                                    'Video': [video_name]})], ignore_index=True)
                    df.to_excel('Motion_detection.xlsx', index=False)
                    counter += 1
                out.release()
                motion_detected_flag = False

        # Show the threshold frame if it is defined
        cv2.imshow('Motion', thresh)

        # Draw the bounding boxes for detected people
        combined_frame = draw_boxes(combined_frame, detections)

        # Show the original frame with bounding boxes
        # cv2.imshow('Webcam', frame)
        cv2.imshow("imgRegion", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()




