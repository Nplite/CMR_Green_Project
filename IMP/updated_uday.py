import cv2
import numpy as np
import datetime
import logging
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from MetalTheft.roi_selector import ROISelector
logging.getLogger('ultralytics').setLevel(logging.WARNING) 
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours

motion_detected_flag = False
start_time = None
counter = 1
object_counter = -1 
roi_1_set, roi_2_set = False, False
roi_1_pts_np, roi_2_pts_np = None, None

# Initialize video capture
cap = cv2.VideoCapture('DATA/17.10.2024.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

roi_selector = ROISelector()
cv2.namedWindow('IP Camera Feed')
cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)

def draw_boxes(frame, detections):
    try:
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if cls == 0:  
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return frame
    except Exception as e:
        print(f"Error in draw_boxes: {e}")


last_motion_time = None
continuous_motion_count = 0

while cap.isOpened():
    try:

        ret, frame = cap.read()
        if not ret:
            break
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Create the ROI mask
        # If ROI 1 is not set, prompt user to select it
        if not roi_1_set:
            cv2.putText(frame, "Select first ROI for motion detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)
            if roi_selector.is_roi_selected():
                roi_1_pts_np = roi_selector.get_roi_points()
                roi_1_set = True
                roi_selector.reset_roi()

        # If ROI 1 is set and ROI 2 is not, prompt user to select ROI 2
        elif not roi_2_set:
            cv2.putText(frame, "Select second ROI for highlighting", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('IP Camera Feed', frame)
            if roi_selector.is_roi_selected():
                roi_2_pts_np = roi_selector.get_roi_points()
                roi_2_set = True
                roi_selector.reset_roi()

        elif roi_1_set and roi_2_set:
            
            mask = np.zeros(frame.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [roi_1_pts_np, roi_2_pts_np], (255, 255, 255))

            mask_inv = cv2.bitwise_not(mask)
            frame_roi = cv2.bitwise_and(frame, mask_inv)
            blurred_roi = cv2.bitwise_and(blurred_frame, mask)
            combined_frame = cv2.add(frame_roi, blurred_roi)
            
            results = model.predict(combined_frame)

            annotator = Annotator(combined_frame)

            detections = []
            person_detected = False
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    c = box.cls
                    if c == 0:  # Check if the detected object is a person (class 0)
                        b = box.xyxy[0]
                        detections.append((b[0], b[1], b[2], b[3], float(box.conf), c))
                        annotator.box_label(b, model.names[int(c)])
                        person_detected = True

            combined_frame = annotator.result()

            # Apply background subtraction to both ROIs
            fgmask = fgbg.apply(combined_frame)
            _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
            mask = np.zeros(fgmask.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [roi_1_pts_np, roi_2_pts_np], (255, 255, 255))
            thresh = cv2.bitwise_and(thresh, mask)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            motion_in_roi = cv2.countNonZero(thresh) > 300
            motion_in_roi_right = cv2.countNonZero(cv2.bitwise_and(thresh, cv2.fillPoly(np.zeros(thresh.shape, dtype=np.uint8), [roi_2_pts_np], (255, 255, 255)))) > 350

            current_time = datetime.datetime.now()

            if motion_in_roi and person_detected:
                if not motion_detected_flag:
                    if last_motion_time is None or (current_time - last_motion_time).total_seconds() > 3:
                        save_snapshot(combined_frame)
                        start_time = current_time
                        motion_detected_flag = True
                        object_counter += 1
                        last_motion_time = current_time
                        continuous_motion_count = 0
                    else:
                        continuous_motion_count += 1
                        if continuous_motion_count > 3:
                            motion_detected_flag = False
                            continuous_motion_count = 0
            else:
                continuous_motion_count = 0
                if motion_detected_flag:
                    end_time = current_time
                    if (end_time - start_time).total_seconds() > 1:
                        counter += 1
                    motion_detected_flag = False

            if motion_in_roi:
                roi_color = (0, 0, 255)  # Red
            else:
                roi_color = (0, 255, 0)  # Green


            motion_mask = np.zeros(combined_frame.shape, dtype=np.uint8)
            cv2.fillPoly(motion_mask, [roi_1_pts_np], roi_color) #wall
            alpha = 0.8
            combined_frame = cv2.addWeighted(combined_frame, alpha, motion_mask, 1 - alpha, 0)
            combined_frame = draw_boxes(combined_frame, detections)


            cv2.imshow('Motion', thresh)
            cv2.imshow("imgRegion", combined_frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):                
            roi_1_set = False
            roi_2_set = False
            roi_selector.reset_roi()


    except Exception as e:
        print(e)

cap.release()
cv2.destroyAllWindows()
