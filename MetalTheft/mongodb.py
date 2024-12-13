import sys
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
from MetalTheft.logger import logging
from MetalTheft.exception import MetalTheptException

class MongoDBHandlerSaving:
    def __init__(self, url="mongodb://localhost:27017", db_name='MetalTheft', snapshot_collection_name='Snapshots', video_collection_name='Videos'):
        try:
            """Initialize the MongoDB client and select the database and collections."""
            self.client = MongoClient(url)
            self.db = self.client[db_name]
            self.snapshot_collection = self.db[snapshot_collection_name]
            self.video_collection = self.db[video_collection_name]
        except Exception as e:
            raise MetalTheptException(e, sys) from e

    def save_snapshot_to_mongodb(self, snapshot_path, date_time, camera_id):
        try:
            """Saves the snapshot path, date, time, and camera_id to MongoDB in the snapshot collection."""
            print("Saving snapshot...")
            date_folder = date_time.strftime('%Y-%m-%d')
            filename = os.path.basename(snapshot_path)
            
            document = {
                'date': date_folder,
                'camera_id': camera_id,
                'images': [{
                    'filename': filename,
                    'path': snapshot_path,
                    'time': date_time.strftime('%H:%M:%S')
                }]
            }
            
            # Update the document if the date and camera_id already exist, otherwise insert a new one
            result = self.snapshot_collection.update_one(
                {'date': date_folder, 'camera_id': camera_id},
                {'$push': {'images': document['images'][0]}},
                upsert=True
            )
            
            if result.upserted_id:
                print(f"Created new document for date: {date_folder} and camera_id: {camera_id}")
            else:
                print(f"Updated existing document for date: {date_folder} and camera_id: {camera_id}")
            
            print(f"Saved snapshot and metadata to MongoDB: {document}")
            logging.info(f"Sending new snapshot document to mongodb for date: {date_folder} and camera_id: {camera_id}")

        except Exception as e:
            raise MetalTheptException(e, sys) from e

    def save_video_to_mongodb(self, video_path, date_time, camera_id):
        try:
            """Saves the video path, date, time, and camera_id to MongoDB in the video collection."""
            print("Saving video...")
            date_folder = date_time.strftime('%Y-%m-%d')
            filename = os.path.basename(video_path)
            
            document = {
                'date': date_folder,
                'camera_id': camera_id,
                'videos': [{
                    'filename': filename,
                    'path': video_path,
                    'time': date_time.strftime('%H:%M:%S')
                }]
            }
            
            # Update the document if the date and camera_id already exist, otherwise insert a new one
            result = self.video_collection.update_one(
                {'date': date_folder, 'camera_id': camera_id},
                {'$push': {'videos': document['videos'][0]}},
                upsert=True
            )
            
            if result.upserted_id:
                print(f"Created new snapshot document for date: {date_folder} and camera_id: {camera_id}")
            else:
                print(f"Updated existing document for date: {date_folder} and camera_id: {camera_id}")
            
            print(f"Saved video and metadata to MongoDB: {document}")
            logging.info(f"Sending new video document to mongodb for date: {date_folder} and camera_id: {camera_id}")

        except Exception as e:
            raise MetalTheptException(e, sys) from e

    # def fetch_snapshots_by_date_and_camera(self, year, month, day, camera_id):
    #     """Fetch snapshots from MongoDB for a specific date and camera_id."""
    #     try:
    #         date_folder = datetime(year, month, day).strftime('%Y-%m-%d')
    #         result = self.snapshot_collection.find_one({'date': date_folder, 'camera_id': camera_id})
    #         logging.info(f"Fetch Snapshot from MongoDB for a specific date:{year}_{month}_{day} and camera_id:{camera_id}")
    #         return result['images'] if result else None
    #     except Exception as e:
    #         raise MetalTheptException(e, sys) from e

    # def fetch_videos_by_date_and_camera(self, year, month, day, camera_id):
    #     """Fetch videos from MongoDB for a specific date and camera_id."""
    #     try:
    #         date_folder = datetime(year, month, day).strftime('%Y-%m-%d')
    #         result = self.video_collection.find_one({'date': date_folder, 'camera_id': camera_id})
    #         logging.info(f"Fetch videos from MongoDB for a specific date:{year}_{month}_{day} and camera_id:{camera_id}")
    #         return result['videos'] if result else None
    #     except Exception as e:
    #         raise MetalTheptException(e, sys) from e

    # def fetch_snapshots_by_month_and_camera(self, year, month, camera_id):
    #     """Fetch snapshots from MongoDB for a specific month and camera_id."""
    #     try:
    #         start_date = datetime(year, month, 1)
    #         # Calculate the end date of the month
    #         if month < 12:
    #             end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    #         else:
    #             end_date = datetime(year + 1, 1, 1) - timedelta(days=1)

    #         # Query for snapshots within the date range and specific camera_id
    #         results = self.snapshot_collection.find({
    #             'camera_id': camera_id,
    #             'date': {
    #                 '$gte': start_date.strftime('%Y-%m-%d'),
    #                 '$lte': end_date.strftime('%Y-%m-%d')
    #             }
    #         })

    #         # Combine snapshots from all results
    #         snapshots = [image for result in results for image in result['images']]
    #         logging.info(f"Fetch Snapshot from MongoDB for a specific month:{year}_{month} and camera_id:{camera_id}")
    #         return snapshots if snapshots else None
    #     except Exception as e:
    #         raise MetalTheptException(e, sys) from e

    # def fetch_videos_by_month_and_camera(self, year, month, camera_id):
    #     """Fetch videos from MongoDB for a specific month and camera_id."""
    #     try:
    #         start_date = datetime(year, month, 1)
    #         # Calculate the end date of the month
    #         if month < 12:
    #             end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    #         else:
    #             end_date = datetime(year + 1, 1, 1) - timedelta(days=1)

    #         # Query for videos within the date range and specific camera_id
    #         results = self.video_collection.find({
    #             'camera_id': camera_id,
    #             'date': {
    #                 '$gte': start_date.strftime('%Y-%m-%d'),
    #                 '$lte': end_date.strftime('%Y-%m-%d')
    #             }
    #         })

    #         # Combine videos from all results
    #         videos = [video for result in results for video in result['videos']]
    #         logging.info(f"Fetch videos from MongoDB for a specific month:{year}_{month} and camera_id:{camera_id}")
    #         return videos if videos else None
    #     except Exception as e:
    #         raise MetalTheptException(e, sys) from e



# @app.get("/videodate/{year}/{month}/{day}", response_model=List[VideoDate])
# async def get_videos(year: int, month: int, day: int, camera_id: int):
#     """Get videos by date."""
#     try:
#         videos = mongo_handler.fetch_videos_by_date_and_camera(year, month, day, camera_id)
#         if videos:
#             return videos
#         raise HTTPException(status_code=404, detail="Videos not found for the given date")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/videomonth/{year}/{month}", response_model=List[VideoMonth])
# async def get_videos_by_month(year: int, month: int, camera_id: int):
#     """Get videos by month."""
#     try:
#         videos = mongo_handler.fetch_videos_by_month_and_camera(year, month, camera_id)
#         if videos:
#             return videos
#         raise HTTPException(status_code=404, detail="No videos found for the given month")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))











import sys
import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
from MetalTheft.logger import logging
from MetalTheft.exception import MetalTheptException


class MongoDBHandlerFetching:
    def __init__(self, url="mongodb://localhost:27017", db_name='MetalTheft', snapshot_collection_name='Snapshots', video_collection_name='Videos'):
        try:
            """Initialize the MongoDB client and select the database and collections."""
            self.client = MongoClient(url)
            self.db = self.client[db_name]
            self.snapshot_collection = self.db[snapshot_collection_name]
            self.video_collection = self.db[video_collection_name]
        except Exception as e:
            raise MetalTheptException(e, sys) from e

    def fetch_snapshots_by_date_and_camera(self, year, month, day, camera_id):
        """Fetch snapshots from MongoDB for a specific date and camera_id."""
        try:
            date_folder = datetime(year, month, day).strftime('%Y-%m-%d')
            result = self.snapshot_collection.find_one({'date': date_folder, 'camera_id': camera_id})
            logging.info(f"Fetch Snapshot from MongoDB for a specific date:{year}_{month}_{day} and camera_id:{camera_id}")
            return result['images'] if result else None
        except Exception as e:
            raise MetalTheptException(e, sys) from e


    def fetch_snapshots_by_month_and_camera(self, year, month, camera_id):
        """Fetch snapshots from MongoDB for a specific month and camera_id."""
        try:
            start_date = datetime(year, month, 1)
            # Calculate the end date of the month
            if month < 12:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)

            # Query for snapshots within the date range and specific camera_id
            results = self.snapshot_collection.find({
                'camera_id': camera_id,
                'date': {
                    '$gte': start_date.strftime('%Y-%m-%d'),
                    '$lte': end_date.strftime('%Y-%m-%d')
                }
            })

            # Combine snapshots from all results
            snapshots = [image for result in results for image in result['images']]
            logging.info(f"Fetch Snapshot from MongoDB for a specific month:{year}_{month} and camera_id:{camera_id}")
            return snapshots if snapshots else None
        except Exception as e:
            raise MetalTheptException(e, sys) from e

