# import logging
# import os
# import boto3
# import sys
# import pytz
# from botocore.exceptions import NoCredentialsError
# from datetime import datetime
# from MetalTheft.exception import MetalTheptException
# from dotenv import load_dotenv

# load_dotenv()

# # Get AWS credentials from environment variables
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')
# bucket_name = os.getenv('bucket_name')

# class AWSConfig:
    
#     def __init__(self):
#         try:
#             # Configure logging
#             logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#             self.logger = logging.getLogger(__name__)

#             # Initialize AWS session
#             self.session = boto3.Session(
#                 aws_access_key_id=AWS_ACCESS_KEY_ID,
#                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#                 region_name=AWS_DEFAULT_REGION
#             )
#             self.logger.info("AWS session initialized")

#         except Exception as e:
#             raise MetalTheptException(e, sys) from e

#     def upload_to_s3(self, file_path, bucket_name=bucket_name, object_name=None):
#         try:
#             """
#             Upload a file to an S3 bucket.

#             :param file_path: Path to the file to upload
#             :param bucket_name: Name of the S3 bucket
#             :param object_name: S3 object name. If not specified, file_name is used
#             :return: URL of the uploaded file if successful, else None
#             """
#             if object_name is None:
#                 object_name = os.path.basename(file_path)

#             self.logger.info(f"Uploading {file_path} to bucket {bucket_name}")
#             s3_client = self.session.client('s3')
#             try:
#                 s3_client.upload_file(file_path, bucket_name, object_name)
#                 url = f"https://{bucket_name}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{object_name}"
#                 self.logger.info(f"File uploaded successfully to {url}")
#                 return url
#             except NoCredentialsError:
#                 self.logger.error("Credentials not available for AWS S3")
#             except Exception as e:
#                 self.logger.error(f"Error uploading file to S3: {e}")
#             return None
        
#         except Exception as e:
#             raise MetalTheptException(e, sys) from e

#     def upload_snapshot_to_s3bucket(self, image_path):
#         try:
#             """
#             Capture an image and upload it to S3 in a date-based folder structure.

#             :param image_path: Path of the image to upload
#             :param bucket_name: Name of the S3 bucket
#             """
#             if image_path and os.path.exists(image_path):
#                 # Get current date in IST
#                 ist = pytz.timezone('Asia/Kolkata')
#                 current_date = datetime.now(ist).strftime('%Y-%m-%d')

#                 # Create S3 object name with date-wise folder structure
#                 file_name = os.path.basename(image_path)
#                 s3_object_name = f"{current_date}/{file_name}"
#                 # Upload to S3 and return the URL


#                 url = self.upload_to_s3(image_path, bucket_name, s3_object_name)

#                 if url:
#                     return url
#                 else:
#                     self.logger.error("Failed to upload image to S3")
#                     return None
#             else:
#                 self.logger.error("Image capture failed or file does not exist")

#         except Exception as e:
#             raise MetalTheptException(e, sys) from e
        

#     def upload_video_to_s3bucket(self, video_path):
#         try:
#             """
#             Capture an video and upload it to S3 in a date-based folder structure.

#             :param video_path: Path of the video to upload
#             :param bucket_name: Name of the S3 bucket
#             """
#             if video_path and os.path.exists(video_path):
#                 # Get current date in IST
#                 ist = pytz.timezone('Asia/Kolkata')
#                 current_date = datetime.now(ist).strftime('%Y-%m-%d')

#                 # Create S3 object name with date-wise folder structure
#                 file_name = os.path.basename(video_path)
#                 s3_object_name = f"{current_date}/{file_name}"
#                 # Upload to S3 and return the URL


#                 url = self.upload_to_s3(video_path, bucket_name, s3_object_name)

#                 if url:
#                     return url
#                 else:
#                     self.logger.error("Failed to upload video to S3")
#                     return None
#             else:
#                 self.logger.error("video capture failed or file does not exist")

#         except Exception as e:
#             raise MetalTheptException(e, sys) from e

# # aws = AWSConfig()
# # aws.upload_to_s3_bucket('snapshots/2024-09-20/16:05:54.jpg')




import logging
import os
import boto3
import sys
import pytz
from botocore.exceptions import NoCredentialsError
from datetime import datetime
from MetalTheft.exception import MetalTheptException
from dotenv import load_dotenv

load_dotenv()

# Get AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')
bucket_name = os.getenv('bucket_name')

class AWSConfig:
    
    def __init__(self):
        try:
            # Configure logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)

            # Initialize AWS session
            self.session = boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_DEFAULT_REGION
            )
            self.logger.info("AWS session initialized")

        except Exception as e:
            raise MetalTheptException(e, sys) from e

    def upload_to_s3(self, file_path, bucket_name=bucket_name, object_name=None):
        try:
            """
            Upload a file to an S3 bucket.

            :param file_path: Path to the file to upload
            :param bucket_name: Name of the S3 bucket
            :param object_name: S3 object name. If not specified, file_name is used
            :return: URL of the uploaded file if successful, else None
            """
            if object_name is None:
                object_name = os.path.basename(file_path)

            self.logger.info(f"Uploading {file_path} to bucket {bucket_name}")
            s3_client = self.session.client('s3')
            try:
                s3_client.upload_file(file_path, bucket_name, object_name)
                url = f"https://{bucket_name}.s3.{AWS_DEFAULT_REGION}.amazonaws.com/{object_name}"
                self.logger.info(f"File uploaded successfully to {url}")
                return url
            except NoCredentialsError:
                self.logger.error("Credentials not available for AWS S3")
            except Exception as e:
                self.logger.error(f"Error uploading file to S3: {e}")
            return None
        
        except Exception as e:
            raise MetalTheptException(e, sys) from e

    def upload_snapshot_to_s3bucket(self, image_path, camera_id):
        try:
            """
            Upload an image to S3 in a date-based and camera_id folder structure.

            :param image_path: Path of the image to upload
            :param camera_id: Identifier of the camera
            """
            if image_path and os.path.exists(image_path):
                # Get current date in IST
                ist = pytz.timezone('Asia/Kolkata')
                current_date = datetime.now(ist).strftime('%Y-%m-%d')

                # Create S3 object name with date-wise and camera_id folder structure
                file_name = os.path.basename(image_path)
                s3_object_name = f"{camera_id}/{current_date}/{file_name}"

                # Upload to S3 and return the URL
                url = self.upload_to_s3(image_path, bucket_name, s3_object_name)

                if url:
                    return url
                else:
                    self.logger.error("Failed to upload image to S3")
                    return None
            else:
                self.logger.error("Image capture failed or file does not exist")

        except Exception as e:
            raise MetalTheptException(e, sys) from e
        

    def upload_video_to_s3bucket(self, video_path, camera_id):
        try:
            """
            Upload a video to S3 in a date-based and camera_id folder structure.

            :param video_path: Path of the video to upload
            :param camera_id: Identifier of the camera
            """
            if video_path and os.path.exists(video_path):
                # Get current date in IST
                ist = pytz.timezone('Asia/Kolkata')
                current_date = datetime.now(ist).strftime('%Y-%m-%d')

                # Create S3 object name with date-wise and camera_id folder structure
                file_name = os.path.basename(video_path)
                s3_object_name = f"{camera_id}/{current_date}/{file_name}"

                # Upload to S3 and return the URL
                url = self.upload_to_s3(video_path, bucket_name, s3_object_name)

                if url:
                    return url
                else:
                    self.logger.error("Failed to upload video to S3")
                    return None
            else:
                self.logger.error("Video capture failed or file does not exist")

        except Exception as e:
            raise MetalTheptException(e, sys) from e

# Example usage:
# aws = AWSConfig()
# aws.upload_snapshot_to_s3bucket('path/to/image.jpg', 'camera_01')
# aws.upload_video_to_s3bucket('path/to/video.mp4', 'camera_01')
