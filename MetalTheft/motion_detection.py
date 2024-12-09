import cv2
import sys
import numpy as np
from PIL import ImageFont
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.utils.utils import save_snapshot, draw_boxes
# from ultralytics.utils.plotting import Annotator
# from ultralytics import YOLO



motion_detected_flag = False

class EuclideanTracker:
    try:
        def __init__(self):
            self.trackers = []  # Store tracked objects (as (x, y, w, h) tuples)

        def update(self, detections):
            try:
                if not self.trackers:
                    self.trackers = detections
                    return self.trackers

                new_trackers = []

                for detection in detections:
                    closest_tracker = None
                    min_distance = float('inf')

                    for tracker in self.trackers:
                        distance = np.linalg.norm(np.array(detection[:2]) - np.array(tracker[:2]))
                        if distance < min_distance:
                            min_distance = distance
                            closest_tracker = tracker

                    if closest_tracker and min_distance < 50:  # Distance threshold to match
                        new_trackers.append(detection)
                        self.trackers.remove(closest_tracker)
                    else:
                        new_trackers.append(detection)

                self.trackers = new_trackers
                return self.trackers
            
            except Exception as e:
                raise MetalTheptException(e, sys) from e
    except Exception as e:
        raise MetalTheptException(e, sys) from e

def detect_motion(frame, blurred_frame, model, fgbg, roi_pts_np):
    try:
        combined_frame = apply_roi_blur(frame, blurred_frame, roi_pts_np)
        results = model.predict(combined_frame)
        # annotator = Annotator(combined_frame, font_size= 6)
        detections = []
        person_detected = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = box.cls
                if c == 0:  # Check if the detected object is a person (class 0)
                    b = box.xyxy[0]
                    detections.append((b[0], b[1], b[2], b[3], float(box.conf), c))
                    # annotator.box_label(b, model.names[int(c)])
                    person_detected = True

        # combined_frame = annotator.result()
        # draw_boxes(combined_frame, detections)

        fgmask = fgbg.apply(combined_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
        _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
        mask = np.zeros(fgmask.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_np], (255, 255, 255))
        thresh = cv2.bitwise_and(thresh, mask)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        return combined_frame, thresh, person_detected, detections
    
    except Exception as e:
        raise MetalTheptException(e, sys) from e

def person_detection_ROI(frame, roi_points, model):
    try:
        """Draw tracked persons with circles and detected persons with rectangles."""
        tracker = EuclideanTracker()
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [roi_points], (255, 255, 255))
        roi_frame = cv2.bitwise_and(frame, mask)
        results = model(roi_frame)

        detections = []
        person_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.cls == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    w, h = x2 - x1, y2 - y1  # Calculate width and height
                    detections.append((x1, y1, w, h))  # Store as (x, y, w, h)
                    cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(roi_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # person_detected = detections
                    person_count += 1

        tracked_objects = tracker.update(detections)
        for (x, y, w, h) in tracked_objects:
            center_x, center_y = int(x + w / 2), int(y + h / 2)  # Calculate center
            cv2.circle(roi_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            # cv2.putText(roi_frame, "Tracked", (center_x - 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return roi_frame , detections, person_count
    except Exception as e:
        raise MetalTheptException(e, sys) from e

def apply_roi_blur(frame, blurred_frame, roi_pts_np):
    try:
        # Create the ROI mask
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts_np], (255, 255, 255))

        # Combine the original frame and the blurred frame using the ROI mask
        mask_inv = cv2.bitwise_not(mask)
        frame_roi = cv2.bitwise_and(frame, mask_inv)
        blurred_roi = cv2.bitwise_and(blurred_frame, mask)
        combined_frame = cv2.add(frame_roi, blurred_roi)

        return combined_frame
    
    except Exception as e:
        raise MetalTheptException(e, sys) from e
    












#############################################################################################################################



# def person_detection_ROI(frame, roi_points, model):       # This is the Backup code
#     tracker = EuclideanTracker()
#     mask = np.zeros_like(frame)
#     cv2.fillPoly(mask, [roi_points], (255, 255, 255))
#     roi_frame = cv2.bitwise_and(frame, mask)
#     results = model(roi_frame)

#     detections = []
#     person_detected = False
#     person_count = 0 

#     for result in results[0].boxes:
#         if result.cls == 0:  
#             x1, y1, x2, y2 = map(int, result.xyxy[0])  # Extract coordinates
#             center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#             detections.append((center_x, center_y, x2 - x1, y2 - y1))
#             person_detected = True
#             person_count += 1

#     tracked_objects = tracker.update(detections)
#     for (x, y, w, h) in tracked_objects:
#         cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
#         cv2.putText(frame, "Person", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     for result in results[0].boxes:
#         if result.cls == 0:  # Draw rectangles only for 'person'
#             x1, y1, x2, y2 = map(int, result.xyxy[0])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     cv2.putText(frame, f"Count: {person_count}", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



#     return frame, person_detected  




# def detect_motion_Roi2(frame, background_subtractor, roi_points, min_contour_area=400):

#     roi_mask = np.zeros_like(frame[:, :, 0])  # Single channel mask (grayscale)
#     cv2.fillPoly(roi_mask, [roi_points], 255)
    
#     foreground_mask = background_subtractor.apply(frame)
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
   
#     foreground_mask_roi = cv2.bitwise_and(foreground_mask, foreground_mask, mask=roi_mask)
#     contours, _ = cv2.findContours(foreground_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         if cv2.contourArea(contour) > min_contour_area:  # Filter small contours
#             x, y, w, h = cv2.boundingRect(contour)
#             # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

#     # Optional: Draw the ROI on the frame for visualization
#     cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=1)

#     return frame, foreground_mask_roi





#############################################################################################################


# def detect_motion(frame, blurred_frame, model, fgbg, roi_pts_np):
#     try:
#         combined_frame = apply_roi_blur(frame, blurred_frame, roi_pts_np)
#         tracker = EuclideanTracker()
#         mask_roi = np.zeros_like(frame)
#         cv2.fillPoly(mask_roi, [roi_pts_np], (255, 255, 255))
#         roi_frame = cv2.bitwise_and(frame, mask_roi)
#         results = model(roi_frame)

#         detections = []
#         person_detected = False
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 if box.cls == 0:  # Person class
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#                     w, h = x2 - x1, y2 - y1  # Calculate width and height
#                     detections.append((x1, y1, w, h))  # Store as (x, y, w, h)
#                     cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(roi_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     person_detected = True

#         tracked_objects = tracker.update(detections)
#         for (x, y, w, h) in tracked_objects:
#             center_x, center_y = int(x + w / 2), int(y + h / 2)  # Calculate center
#             cv2.circle(roi_frame, (center_x, center_y), 5, (255, 0, 0), -1)

#         fgmask = fgbg.apply(combined_frame)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
#         _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
#         mask = np.zeros(fgmask.shape, dtype=np.uint8)
#         cv2.fillPoly(mask, [roi_pts_np], (255, 255, 255))
#         thresh = cv2.bitwise_and(thresh, mask)
#         kernel = np.ones((5, 5), np.uint8)
#         thresh = cv2.dilate(thresh, kernel, iterations=2)

#         return combined_frame, thresh, person_detected, detections
    
#     except Exception as e:
#         raise MetalTheptException(e, sys) from e

