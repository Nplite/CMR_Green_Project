from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import Dict, Optional
import cv2
import numpy as np
import sys
import yaml
from typing import List
import torch
import threading
import time
import logging
from collections import deque
from datetime import datetime
from fastapi.responses import JSONResponse
import json
import cv2
import numpy as np
import sys
import yaml
import torch
import threading
import time
import logging
from collections import deque
from datetime import datetime
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours
from MetalTheft.motion_detection import detect_motion, person_detection_ROI3
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from ultralytics import YOLO
from threading import Thread, Lock
import queue
logging.getLogger('ultralytics').setLevel(logging.WARNING) 


# Import your existing classes (assuming they're in separate files)
from MetalTheft.constant import *
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot, save_video, draw_boxes, draw_motion_contours
from MetalTheft.motion_detection import detect_motion, person_detection_ROI3
from MetalTheft.roi_selector import ROISelector
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.aws import AWSConfig
from ultralytics import YOLO

mongo_handler = MongoDBHandler()

# Create FastAPI app
app = FastAPI(title="Camera Surveillance System API")

# Pydantic models for request validation
class CameraURL(BaseModel):
    camera_id: int
    url: str

class CameraROIReset(BaseModel):
    camera_id: int

# Global instance of MultiCameraSystem
camera_system = None

# Pydantic Models
class SnapDate(BaseModel):
    filename: str
    path: str
    time: str


class SnapMonth(BaseModel):
    filename: str
    path: str
    time: str


class CameraStream:
    def __init__(self, rtsp_url, camera_id):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(rtsp_url)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)  # Default to 30 FPS if unknown
        self.frame_queue = queue.Queue(maxsize=30)
        self.stopped = False
        self.lock = Lock()
        
    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                return

            with self.lock:
                if not self.frame_queue.full():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        continue

                    # Clear stale frames to keep the queue fresh
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break

                    self.frame_queue.put(frame)

            time.sleep(1 / self.fps)  #
                
    def read(self):
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, None
    
    def stop(self):
        self.stopped = True
        with self.lock:
            self.cap.release()

class CameraProcessor:
    def __init__(self, camera_id, rtsp_url):
        self.camera_id = camera_id
        self.stream = CameraStream(rtsp_url, camera_id)
        self.roi_1_set = False
        self.roi_2_set = False
        self.roi_3_set = False
        self.roi_1_pts_np = None
        self.roi_2_pts_np = None
        self.roi_3_pts_np = None
        self.alpha = 0.6
        self.counter = 1
        self.last_motion_time = None
        self.current_time = None
        self.start_time = None
        self.motion_direction_window = 1
        self.motion_detected_flag = False
        self.roi1_motion_time = None
        self.roi2_motion_time = None
        self.recording_after_motion = False
        self.recording_end_time = None
        self.video_path  = None
        self.snapshot_path = None
        self.is_processing = False
        self.setup_complete = False
        self.object_counter = -1 
        self.email = EmailSender()
        self.mongo_handler = MongoDBHandler()
        self.aws = AWSConfig()

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = YOLO('yolov8n.pt').to(device)
        # print(f"Model loaded on: {device}")

        self.model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.engine', task = 'detect', verbose=True)
        
        # Initialize components
        self.roi_selector = ROISelector()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=True)
        
        # Initialize motion buffer
        self.motion_buffer_duration = 5
        self.motion_buffer_fps = 30
        self.motion_frame_buffer = deque(maxlen=self.motion_buffer_fps * self.motion_buffer_duration)
        self.current_people_in_roi3 = 0
        
        # Video writer setup
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_motion = None
        
        # Window name
        self.window_name = f'Camera {self.camera_id}'

    def setup_roi(self):
        """Handle ROI setup for this camera"""
        print(f"\nSetting up ROIs for Camera {self.camera_id}")
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.roi_selector.select_point)

        while not (self.roi_1_set and self.roi_2_set and self.roi_3_set):
            ret, frame = self.stream.read()
            if not ret:
                continue

            if not self.roi_1_set:
                cv2.putText(frame, f"Select first ROI for camera {self.camera_id} (Click 4 points)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.roi_selector.is_roi_selected():
                    self.roi_1_pts_np = self.roi_selector.get_roi_points()
                    self.roi_1_set = True
                    self.roi_selector.reset_roi()
                    print(f"Camera {self.camera_id}: First ROI set")
                    time.sleep(0.5)  # Small delay for visual feedback

            elif not self.roi_2_set:
                cv2.putText(frame, f"Select second ROI for camera {self.camera_id} (Click 4 points)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.roi_selector.is_roi_selected():
                    self.roi_2_pts_np = self.roi_selector.get_roi_points()
                    self.roi_2_set = True
                    self.roi_selector.reset_roi()
                    print(f"Camera {self.camera_id}: Second ROI set")
                    time.sleep(0.5)  # Small delay for visual feedback

            elif not self.roi_3_set:
                cv2.putText(frame, f"Select Third ROI for camera {self.camera_id} (Click 4 points)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.roi_selector.is_roi_selected():
                    self.roi_3_pts_np = self.roi_selector.get_roi_points()
                    self.roi_3_set = True
                    self.roi_selector.reset_roi()
                    print(f"Camera {self.camera_id}: First ROI set")
                    time.sleep(0.5)  # Small delay for visual feedback

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('r'):
                self.reset_roi()
                
        self.setup_complete = True
        print(f"Camera {self.camera_id} setup complete!")
        return True

    def reset_roi(self):
        self.roi_1_set = False
        self.roi_2_set = False
        self.roi_3_set = False
        self.roi_selector.reset_roi()
        self.setup_complete = False
        print(f"Camera {self.camera_id}: ROI reset. Please set ROIs again.")
        # self.setup_roi()  # Trigger ROI setup only for this camera

    def reset_all_roi(self):
        self.roi_1_set = False
        self.roi_2_set = False
        self.roi_3_set = False
        self.roi_selector.reset_roi()
        self.setup_complete = False

    def process_frame(self, frame, model):
        if not self.setup_complete:
            return frame, None

        try:
            combined_frame1 = frame.copy()
            combined_frame2 = frame.copy()

            self.motion_frame_buffer.append(frame.copy())
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

            # Process ROI 1
            combined_frame1, thresh_ROI1, person_detected, detections = detect_motion(frame, blurred_frame, model, self.fgbg, self.roi_1_pts_np)
            motion_in_roi1 = cv2.countNonZero(thresh_ROI1) > 100
            motion_mask_1 = np.zeros_like(combined_frame1)
            cv2.fillPoly(motion_mask_1, [self.roi_1_pts_np], (0, 255, 0))
            combined_frame1 = cv2.addWeighted(combined_frame1, 0.6, motion_mask_1, 0.4, 0)
            
            # Process ROI 2
            combined_frame2, thresh_ROI2, _, _ = detect_motion(frame, blurred_frame, model, self.fgbg, self.roi_2_pts_np)
            motion_in_roi2 = cv2.countNonZero(thresh_ROI2) > 350
            motion_mask_2 = np.zeros_like(frame)
            cv2.fillPoly(motion_mask_2, [self.roi_2_pts_np], (0,0,0))  
            combined_frame2 = cv2.addWeighted(combined_frame2, 0.6, motion_mask_2, 0.4, 0)

            # Process ROI 2
            combined_frame3, person_in_roi3, person_counter = person_detection_ROI3(frame=frame, roi_points= self.roi_3_pts_np, model= self.model)
            cv2.polylines(combined_frame3, [self.roi_3_pts_np], isClosed=True, color=(0, 255, 0), thickness=1)
            self.current_people_in_roi3 = len(person_in_roi3)
            

            combined_frame = self.process_motion(frame,
                combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, person_counter,
                motion_in_roi2, person_detected, person_in_roi3, thresh_ROI1, thresh_ROI2, detections)
            
            return combined_frame, thresh_ROI1
            
        except Exception as e:
            print(f"Error processing camera {self.camera_id}: {str(e)}")
            return frame, None

    def process_motion(self, frame, combined_frame1, combined_frame2, combined_frame3, motion_in_roi1, 
                      person_counter, motion_in_roi2, person_detected, person_in_roi3, thresh_ROI1, thresh_ROI2, detections):
        # Draw ROIs
        cv2.polylines(combined_frame1, [self.roi_1_pts_np], True, (0, 255, 0), 1)
        cv2.polylines(combined_frame2, [self.roi_2_pts_np], True, (0, 255, 0), 1)
        
        # Process motion in ROI 1 & ROI2
        if motion_in_roi1:
            self.roi1_motion_time = time.time()
            draw_motion_contours(combined_frame1, thresh_ROI1, self.roi_1_pts_np)
       
        if motion_in_roi2:
            self.roi2_motion_time = time.time()
            draw_motion_contours(combined_frame2, thresh_ROI2, self.roi_2_pts_np)

        # combined_frame = cv2.add(combined_frame1, combined_frame2)
        combined_frame = cv2.add(cv2.add(combined_frame1, combined_frame2), combined_frame3)

        # Handle recording logic
        self.current_time = datetime.now()
        if (motion_in_roi1 and person_detected):
            if self.last_motion_time is None or (self.current_time - self.last_motion_time).total_seconds() > 3:
                if self.roi2_motion_time is not None and (self.roi1_motion_time - self.roi2_motion_time) <= self.motion_direction_window:
                    motion_in_roi2_to_roi1 = True  # Direction is confirmed from ROI2 to ROI1 
                    roi_color = (0, 0, 255) if motion_in_roi2_to_roi1 else (0, 255, 0)
                    cv2.fillPoly(combined_frame, [self.roi_1_pts_np], roi_color)  
                    self.last_motion_time = self.current_time  
                    self.object_counter += 1

                else:
                    motion_in_roi2_to_roi1 = False 

        if (motion_in_roi1 and person_detected) or person_in_roi3:
            if (self.roi2_motion_time is not None and (self.roi1_motion_time - self.roi2_motion_time) <= self.motion_direction_window) or person_in_roi3:
                if not self.motion_detected_flag:
                    self.video_path = save_video()
                    self.out_motion = cv2.VideoWriter(self.video_path, self.fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                    self.snapshot_path = save_snapshot(combined_frame)
                    while self.motion_frame_buffer:
                        self.out_motion.write(self.motion_frame_buffer.popleft())
                    self.motion_detected_flag = True
                    self.recording_after_motion = False
                    self.motion_frame_buffer.clear()  # Clear the buffer

                # Write current frame
                self.out_motion.write(combined_frame)

        elif self.motion_detected_flag:
            if not self.recording_after_motion:
                self.recording_end_time = time.time() + 5 
                self.recording_after_motion = True

            if time.time() <= self.recording_end_time:
                self.out_motion.write(combined_frame)
            else:
                self.motion_detected_flag = False
                self.recording_after_motion = False
                self.out_motion.release()
                start_time = datetime.now()

                video_url = self.aws.upload_video_to_s3bucket(self.video_path)
                threading.Thread(target=self.mongo_handler.save_video_to_mongodb, args=(video_url, start_time))
                threading.Thread(target=self.email.send_alert_email, args=(self.snapshot_path, video_url, self.camera_id)).start()
                self.counter += 1
        
        # Draw detection boxes
        cv2.putText(combined_frame, f'Object Count: {self.object_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(combined_frame, f'Person Count: {person_counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        draw_boxes(combined_frame, detections)
        
        return combined_frame

class MultiCameraSystem:
    def __init__(self):
        self.camera_processors = {}
        self.model = YOLO('/home/alluvium/Desktop/Namdeo/CMR_Project/yolov8n.engine', verbose=True, task='detect')
        self.is_running = False
        self.processing_thread = None

    def setup_cameras(self):
        """Set up ROIs for each camera sequentially"""
        for camera_id, processor in self.camera_processors.items():
            if not processor.setup_complete:
                if not processor.setup_roi():
                    return False
        return True

    async def add_camera(self, camera_id: int, url: str) -> dict:
        """Add a new camera to the system"""
        try:
            if camera_id in self.camera_processors:
                raise HTTPException(status_code=400, detail="Camera ID already exists")
            
            processor = CameraProcessor(camera_id, url)
            processor.stream.start()
            self.camera_processors[camera_id] = processor
            
            # Start the processing thread if it's not already running
            if not self.is_running:
                self.start_processing()
            
            return {"status": "success", "message": f"Camera {camera_id} added successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_camera_count(self) -> int:
        """Get the total number of cameras in the system"""
        return len(self.camera_processors)

    # def get_object_counts(self, camera_id: int ) -> dict:
        """Get object counts for all cameras"""
        object_counts = {}
        for camera_id, processor in self.camera_processors.items():
            object_counts[camera_id] = {
                "object_count": processor.object_counter,
            }
        return object_counts

    # def get_people_counts(self, camera_id: int ) -> dict:
        """Get object counts for all cameras"""
        people_counts = {}
        for camera_id, processor in self.camera_processors.items():
            people_counts[camera_id] = {
                "person_count": processor.person_counter if hasattr(processor, 'person_counter') else 0
            }
        return people_counts

    def get_object_counts(self, camera_id: Optional[int] = None) -> dict:
        """Get object counts for all cameras or a specific camera"""
        if camera_id is not None:
            if camera_id in self.camera_processors:
                processor = self.camera_processors[camera_id]
                return {camera_id: {"object_count": processor.object_counter}}
            else:
                raise HTTPException(status_code=404, detail=f"Camera ID {camera_id} not found")
        
        # If no specific camera_id is provided, return counts for all cameras
        object_counts = {
            cam_id: {"object_count": processor.object_counter}
            for cam_id, processor in self.camera_processors.items()
        }
        return object_counts

    def get_people_counts(self, camera_id: Optional[int] = None) -> dict:
        """Get the current number of people in ROI3 for a specific camera"""
        if camera_id is not None:
            if camera_id in self.camera_processors:
                processor = self.camera_processors[camera_id]
                return {camera_id: {"current_people_in_roi3": processor.current_people_in_roi3}}
            else:
                raise HTTPException(status_code=404, detail=f"Camera ID {camera_id} not found")

        # If no specific camera_id is provided, return counts for all cameras
        people_counts = {
            cam_id: {"current_people_in_roi3": processor.current_people_in_roi3}
            for cam_id, processor in self.camera_processors.items()
        }
        return people_counts

    def reset_all_rois(self):
        for processor in self.camera_processors.values():
            processor.reset_roi()

    async def reset_camera_roi(self, camera_id: int) -> dict:
        """Reset ROI for a specific camera"""
        if camera_id not in self.camera_processors:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        try:
            processor = self.camera_processors[camera_id]
            # processor.reset_roi()
            processor.reset_all_roi()
            # return {"status": "success", "message": f"ROI reset for camera {camera_id}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def start_processing(self):
        """Start processing all cameras in a separate thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_cameras)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_cameras(self):
        """Main processing loop for all cameras"""
        if not self.setup_cameras():
            print("Camera setup incomplete. Exiting...")
            return

        print("\nAll cameras configured! Starting monitoring...")
        
        while self.is_running:
            try:
                for camera_id, processor in self.camera_processors.items():
                    ret, frame = processor.stream.read()
                    if not ret:
                        continue
                    
                    processed_frame, motion = processor.process_frame(frame, self.model)
                    cv2.imshow(processor.window_name, processed_frame)

                key = cv2.waitKey(int(1000 / 30)) & 0xFF 
                if key == ord('q'):
                    break

                # elif key == ord('r'):
                #     camera_id = int(input("Enter Camera ID to reset ROI: "))
                #     self.reset_camera_roi(camera_id)

                elif key == ord('a'):
                    print("\nResetting all ROIs...")
                    self.reset_all_rois()
                    if not self.setup_cameras():
                        break  
                    
                    
            except Exception as e:
                logging.error(f"Error in camera processing: {str(e)}")
                continue

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        for processor in self.camera_processors.values():
            processor.stream.stop()
        cv2.destroyAllWindows()

camera_system = MultiCameraSystem()


# FastAPI Routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Metal Theft Detection System"}

# API endpoints
@app.post("/cameras/add")
async def add_camera(camera_data: CameraURL):
    """Add a new camera to the system"""
    return await camera_system.add_camera(camera_data.camera_id, camera_data.url)

@app.get("/cameras/count")
async def get_camera_count():
    """Get the total number of cameras in the system"""
    count = camera_system.get_camera_count()
    return {"camera_count": count}

# @app.get("/cameras/object-counts")
# async def get_object_counts():
#     """Get object counts for all cameras"""
#     counts = camera_system.get_object_counts()
#     return {"counts": counts}

@app.get("/cameras/object-count/{camera_id}")
async def get_object_count(camera_id: int):
    """Get object count for a specific camera"""
    try:
        count = camera_system.get_object_counts(camera_id=camera_id)
        return {"counts": count}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

@app.get("/cameras/people-count/{camera_id}")
async def get_people_count(camera_id: int):
    """Get the current number of people in ROI3 for a specific camera"""
    try:
        count = camera_system.get_people_counts(camera_id=camera_id)
        return {"counts": count}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})


@app.post("/cameras/reset-roi")
async def reset_camera_roi(reset_data: CameraROIReset):
    """Reset ROI for a specific camera"""
    return await camera_system.reset_camera_roi(reset_data.camera_id)


@app.get("/cameras/working")
async def get_working_cameras():
    """Get number of working cameras"""
    try:
        if camera_system is None:
            return {"working_cameras": 0}
        
        working_count = sum(
            1 for processor in camera_system.camera_processors.values()
            if not processor.stream.stopped
        )
        return {
            "working_cameras": working_count,
            "total_cameras": len(camera_system.camera_processors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup when shutting down"""    
    if camera_system:
        camera_system.cleanup()

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        if camera_system:
            camera_system.cleanup()

            

