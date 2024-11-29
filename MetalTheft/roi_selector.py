import cv2
import sys
import numpy as np
from MetalTheft.exception import MetalTheptException


class ROISelector:
    try:
        def __init__(self):
            self.points = []
            self.roi_selected = False

        def select_point(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.points) < 4:
                    self.points.append((x, y))
                    print(f"Point {len(self.points)} selected: ({x}, {y})")
                    if len(self.points) == 4:
                        self.roi_selected = True
                        self.draw_roi(param)

        def draw_roi(self, image):
            pts = np.array(self.points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        def reset_roi(self):
            self.points = []
            self.roi_selected = False
            print("ROI has been reset. You can select new points.")

        def get_roi_points(self):
            return np.array(self.points, np.int32).reshape((-1, 1, 2))

        def is_roi_selected(self):
            return self.roi_selected
    
    except Exception as e:
        raise MetalTheptException(e, sys) from e