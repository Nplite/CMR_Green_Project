from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from MetalTheft.mongodb import MongoDBHandler
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING) 
from MetalTheft.constant import *   
from MetalTheft.mongodb import MongoDBHandler
from app import CameraProcessor, MultiCameraSystem, CameraStream
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Camera Surveillance System API")
# Handle CORS protection
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*","Authorization", "Content-Type"],
)

class CameraURL(BaseModel):
    camera_id: int
    url: str
class CameraROIReset(BaseModel):
    camera_id: int
    object_id: int
class SnapDate(BaseModel):
    filename: str
    path: str
    time: str
class SnapMonth(BaseModel):
    filename: str
    path: str
    time: str
class Object_Count(BaseModel):
    camera_id : int
    object_count: int
class ObjectCountsResponse(BaseModel):
    counts: List[Object_Count]
class People_Count(BaseModel):
    camera_id: int
    people_count: int
class PeopleCountsResponse(BaseModel):
    counts: List[People_Count]
camera_system = None
camera_system = MultiCameraSystem()

mongo_handler = MongoDBHandler()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Metal Theft Detection System"}

# API endpoints
@app.post("/cameras/add")
async def add_camera(camera_data: CameraURL):
    """Add a new camera to the system"""
    return await camera_system.add_camera(camera_data.camera_id, camera_data.url)

# @app.post("/cameras/reset-roi")
# async def reset_camera_roi(reset_data: CameraROIReset):
#     """Reset ROI for a specific camera"""
#     return await camera_system.reset_camera_roi(reset_data.camera_id)


@app.get("/cameras/count")
async def get_camera_count():
    """Get the total number of cameras in the system"""
    count = camera_system.get_camera_count()
    return {"camera_count": count}

# @app.get("/cameras/object-counts/{camera_id}", response_model=ObjectCountsResponse)
# async def total_object_counts(camera_id: int):
#     raw_object_counts = camera_system.get_object_counts(camera_id)
#     object_counts_list = [
#         {"camera_id": cam_id, "object_count": counts["object_count"]}
#         for cam_id, counts in raw_object_counts.items()    ]
#     return {"counts": object_counts_list}


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

@app.get("/snapdate/{year}/{month}/{day}", response_model=List[SnapDate])
async def get_snapshots(year: int, month: int, day: int):
    """Get snapshots by date."""
    try:
        snapshots = mongo_handler.fetch_snapshots_by_date(year, month, day)
        if snapshots:
            return snapshots
        raise HTTPException(status_code=404, detail="Snapshots not found for the given date")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/snapmonth/{year}/{month}", response_model=List[SnapMonth])
async def get_snapshots_by_month(year: int, month: int):
    """Get snapshots by month."""
    try:
        snapshots = mongo_handler.fetch_snapshots_by_month(year, month)
        if snapshots:
            return snapshots
        raise HTTPException(status_code=404, detail="No snapshots found for the given month")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

            

