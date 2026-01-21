import os
import uuid
import cv2
import glob
import numpy as np
from PIL import Image

class ImageStorage:
    def __init__(self):
        self.crop_dir = os.getenv("CROP_IMAGE_DIR")
        self.origin_dir = os.getenv("ORIGIN_IMAGE_DIR")
        self._ensure_dir(self.crop_dir)
        self._ensure_dir(self.origin_dir)

    def _ensure_dir(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Directory created: {path}")

    def get_origin_image(self, request_id: str):
        search_pattern = os.path.join(self.origin_dir, f"{request_id}.*")
        matching_files = glob.glob(search_pattern)

        if not matching_files:
            print(f"Warning: No file found starting with {request_id}")
            return None

        image_path = matching_files[0]
        try:
            image = Image.open(image_path)
            print(f"Success: Loaded image from {image_path}")
            return image
        except Exception as e:
            print(f"Error: Cannot open image {image_path}. {e}")
            return None

    def save_crop(self, img: np.ndarray, request_id: str) -> str:
        target_dir = os.path.join(self.crop_dir, request_id)
        self._ensure_dir(target_dir)
        
        filename = f"{uuid.uuid4().hex[:8]}.jpg"
        file_path = os.path.join(target_dir, filename)
        
        success = cv2.imwrite(file_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise IOError(f"Failed to save image: {file_path}")
            
        return filename