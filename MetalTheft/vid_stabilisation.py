
import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self):
        self.prev_gray = None
        self.transforms = []
        self.smooth_transforms = []
        self.frame_count = 0

    def stabilised_frame(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize the previous frame if it's the first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        # Calculate optical flow between the previous and the current frame
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute the transformation matrix
        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])
        transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        # Apply the transformation to the current frame
        stabilized_frame = cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))

        # Update the previous frame
        self.prev_gray = gray

        return stabilized_frame

    def reset(self):
        # Reset all variables for a new video
        self.prev_gray = None
        self.transforms = []
        self.smooth_transforms = []
        self.frame_count = 0