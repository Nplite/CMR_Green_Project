import cv2
import numpy as np
import winsound
import threading
import pandas as pd
import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the Excel file
df = pd.read_excel('Motion_detection.xlsx', sheet_name='Sheet1')

motion_detected_flag = False
start_time = None
counter = 1
object_counter = -1 

# Initialize video capture
cap = cv2.VideoCapture('./Videos/BW.mp4')
mask = cv2.imread('./Videos/mask3.png', 0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Get input video frame width, height, and fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output file name and codec
output_file = 'metal.avi'
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
roi_pts = [850, 3, 850, 875, 540, 865, 780, 1]  # ROI 1
roi_pts_right = [880,712,861,123,1278,189,1262,712]  # ROI 2
roi_pts_gray1 = [2,0,730,7,445,715,1,710]  # Gray Area 1
roi_pts_gray2 = [853,2,1278,5,1271,174,863,111]  # Gray Area 2

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

def play_alarm():
    winsound.Beep(2500, 1000)

def calculate_centroid(points):
    points = np.array(points).reshape((-1, 2))
    centroid = np.mean(points, axis=0)
    return int(centroid[0]), int(centroid[1])

last_motion_time = None
continuous_motion_count = 0

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize mask to match frame dimensions
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_resized_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        blurred_frame = cv2.GaussianBlur(frame, (31, 31), 0)
        
        # Create the ROI mask
        roi_pts_np = np.array(roi_pts, np.int32)
        roi_pts_np = roi_pts_np.reshape((-1, 1, 2))
        
        roi_pts_right_np = np.array(roi_pts_right, np.int32)
        roi_pts_right_np = roi_pts_right_np.reshape((-1, 1, 2))
        
        roi_pts_gray1_np = np.array(roi_pts_gray1, np.int32)
        roi_pts_gray1_np = roi_pts_gray1_np.reshape((-1, 1, 2))
        
        roi_pts_gray2_np = np.array(roi_pts_gray2, np.int32)
        roi_pts_gray2_np = roi_pts_gray2_np.reshape((-1, 1, 2))
        
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_np, roi_pts_right_np, roi_pts_gray1_np, roi_pts_gray2_np], (255, 255, 255))

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
                    annotator.box_label(b, label="worker")

        combined_frame = annotator.result()

        # Apply background subtraction to both ROIs
        fgmask = fgbg.apply(combined_frame)
        _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
        mask = np.zeros(fgmask.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_np, roi_pts_right_np], (255, 255, 255))
        thresh = cv2.bitwise_and(thresh, mask)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Check if there is motion in each ROI
        motion_in_roi = cv2.countNonZero(thresh) > 600
        motion_in_roi_right = cv2.countNonZero(cv2.bitwise_and(thresh, cv2.fillPoly(np.zeros(thresh.shape, dtype=np.uint8), [roi_pts_right_np], (255, 255, 255)))) > 350

        current_time = datetime.datetime.now()

        if motion_in_roi or motion_in_roi_right:
            if not motion_detected_flag:
                if last_motion_time is None or (current_time - last_motion_time).total_seconds() > 3:
                    threading.Thread(target=play_alarm).start()
                    start_time = current_time
                    video_name = 'motion_' + str(counter) + '.mp4'
                    out_motion = cv2.VideoWriter(video_name, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
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
                    df = pd.concat([df, pd.DataFrame({'Start Time': [start_time.strftime('%d-%m-%Y %H:%M:%S')],
                                                    'End Time': [end_time.strftime('%d-%m-%Y %H:%M:%S')],
                                                    'Video': [video_name]})], ignore_index=True)
                    df.to_excel('Motion_detection.xlsx', index=False)
                    counter += 1
                out_motion.release()
                motion_detected_flag = False

        # Create a mask image filled with the appropriate color
        if motion_in_roi:
            roi_color = (0, 0, 255)  # Red
        else:
            roi_color = (0, 255, 0)  # Green

        if motion_in_roi_right:
            roi_right_color = (0, 0, 255)  # Red
        else:
            roi_right_color = (250,0,0)  # Blue

        gray_color = (0, 165, 255)  # Gray color for the gray areas

        motion_mask = np.zeros(combined_frame.shape, dtype=np.uint8)
        cv2.fillPoly(motion_mask, [roi_pts_np], roi_color) #wall
        cv2.fillPoly(motion_mask, [roi_pts_right_np], roi_right_color) #land
        cv2.fillPoly(motion_mask, [roi_pts_gray1_np, roi_pts_gray2_np], gray_color) #factory
        cv2.fillPoly(motion_mask, [roi_pts_gray2_np], gray_color)#sky

        # Apply the mask to the frame using cv2.addWeighted()
        alpha = 0.8
        combined_frame = cv2.addWeighted(combined_frame, alpha, motion_mask, 1 - alpha, 0)

        # Draw the bounding boxes for detected people
        combined_frame = draw_boxes(combined_frame, detections)
        cv2.putText(combined_frame, f'Object Count: {object_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Calculate centroids and add titles for each ROI
        roi1_centroid = calculate_centroid(roi_pts)
        roi2_centroid = calculate_centroid(roi_pts_right)
        gray1_centroid = calculate_centroid(roi_pts_gray1)
        gray2_centroid = calculate_centroid(roi_pts_gray2)
        
        cv2.putText(combined_frame, 'Wall', (roi1_centroid[0] - 40, roi1_centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'Land', (roi2_centroid[0] - 40, roi2_centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'Factory', (gray1_centroid[0] - 80, gray1_centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, 'Sky', (gray2_centroid[0] - 80, gray2_centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if motion_detected_flag:
            out_motion.write(combined_frame)  # Write frame to motion video file
        
        out.write(combined_frame)  # Write frame to general video file

        cv2.imshow('Motion', thresh)

        # Show the original frame with bounding boxes
        cv2.imshow("imgRegion", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)

cap.release()
out.release()
cv2.destroyAllWindows()
