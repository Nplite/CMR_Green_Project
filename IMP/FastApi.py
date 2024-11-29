import cv2
import sys
import numpy as np
import threading
from typing import List
from ultralytics import YOLO
from datetime import datetime
from pydantic import BaseModel
from MetalTheft.constant import *
from MetalTheft.aws import AWSConfig
from MetalTheft.mongodb import MongoDBHandler
from MetalTheft.roi_selector import ROISelector
from MetalTheft.exception import MetalTheptException
from MetalTheft.send_email import EmailSender
from MetalTheft.utils.utils import save_snapshot
from MetalTheft.motion_detection import detect_motion
from camera_app import FastAPI, BackgroundTasks, HTTPException, Path



app = FastAPI()


motion_detected_flag = False
start_time = None
counter = 1
cap = None 
fgbg = cv2.createBackgroundSubtractorMOG2()


email = EmailSender()
roi_selector = ROISelector()
mongo_handler = MongoDBHandler()
aws = AWSConfig()
model = YOLO('yolov8n.pt')

class RTSPInput(BaseModel):
    rtsp_url: str

def process_video_feed(rtsp_url: str):
    global motion_detected_flag, start_time, counter, cap
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error: Could not open RTSP stream.")

    cv2.namedWindow('IP Camera Feed')
    cv2.setMouseCallback('IP Camera Feed', roi_selector.select_point, param=None)

    while cap.isOpened():
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

                combined_frame, thresh, person_detected = detect_motion(
                    frame, blurred_frame, model, fgbg, roi_pts_np)

                motion_in_roi = cv2.countNonZero(thresh) > 400

                if motion_in_roi and person_detected:
                    if not motion_detected_flag:
                        snapshot_path = save_snapshot(combined_frame)
                        if snapshot_path:
                            threading.Thread(target=email.send_alert_email, args=(snapshot_path,)).start()
                            current_time = datetime.now()
                            snapshot_url = aws.upload_to_s3_bucket(snapshot_path)
                            mongo_handler.save_to_mongo(snapshot_url, current_time)

                        start_time = current_time
                        motion_detected_flag = True

                else:
                    if motion_detected_flag:
                        end_time = datetime.now()
                        if (end_time - start_time).total_seconds() > 1:
                            counter += 1
                        motion_detected_flag = False

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
            elif key == ord('r'):
                roi_selector.reset_roi()

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise MetalTheptException(e, sys) from e

    cap.release()
    cv2.destroyAllWindows()




class SnapDate(BaseModel):
    filename: str
    path: str
    time: str



class SnapMonth(BaseModel):
    filename: str
    path: str
    time: str



@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Metal Theft Detection System"}



@app.post("/start-detection")
def start_detection(rtsp_input: RTSPInput, background_tasks: BackgroundTasks):
    global cap
    if cap is None or not cap.isOpened():
        background_tasks.add_task(process_video_feed, rtsp_input.rtsp_url)
        return {"message": f"Motion detection started for RTSP URL: {rtsp_input.rtsp_url}"}
    else:
        raise HTTPException(status_code=400, detail="Motion detection is already running")



@app.post("/stop-detection")
def stop_detection():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        return {"message": "Motion detection stopped"}
    else:
        raise HTTPException(status_code=400, detail="No active motion detection to stop")



@app.get("/status")
def get_status():
    global cap
    if cap is not None and cap.isOpened():
        return {"message": "Motion detection is running"}
    else:
        return {"message": "No active motion detection"}
    


@app.get("/snapdate/{year}/{month}/{day}", response_model=List[SnapDate])
async def get_snapshots(year: int, month: int, day: int):
    try:
        snapshots = mongo_handler.fetch_snapshots_by_date(year, month, day)

        if snapshots:
            return snapshots
        else:
            raise HTTPException(status_code=404, detail="Snapshots not found for the given date")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/snapmonth/{year}/{month}", response_model=List[SnapMonth])
async def get_snapshots_by_month(year: int, month: int):
    try:
        snapshots = mongo_handler.fetch_snapshots_by_month(year, month)
        if snapshots:
            return snapshots
        else:
            raise HTTPException(status_code=404, detail="No snapshots found for the given month")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# @app.get("/snapdate/{year}/{month}/{day}", response_model=List[SnapDate])
# async def get_snapshots(year: int, month: int, day: int):
#     try:
#         snapshots = mongo_handler.fetch_snapshots_by_date(year, month, day)

#         if snapshots:
#             # Format snapshot details as a string for the email
#             snapshot_details = "\n".join([str(snapshot) for snapshot in snapshots])

#             try:
#                 result = email.daily_report_email(snapshot_details)  
#                 return {"message": result, "snapshots": snapshots}
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=f"Email sending failed: {str(e)}")
#         else:
#             raise HTTPException(status_code=404, detail="Snapshots not found for the given date")
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)







