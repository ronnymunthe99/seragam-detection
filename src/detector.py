import sys, os
import torch
from configparser import ConfigParser
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import urllib.request
import base64
from PIL import Image
from typing import List, Dict, Union

# Set path
path_this = os.path.realpath(os.path.dirname(__file__))
path_project = os.path.realpath(os.path.join(path_this, ".."))
sys.path.append(path_this)
sys.path.append(path_project)

# Optional imports if needed
try:
    from src.helper_classes import color_mapping  # Keep only color_mapping
except ImportError:
    color_mapping = {
        "black": [0, 0, 0],
        "white": [255, 255, 255],
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0],
        "gray": [128, 128, 128]
    }

class ObjectDetectionModel:
    def __init__(
        self,
        root_dir: str = None,
        yolo_model_seragam_path: str = None,
        detector_type: str = "Detector",
        device: str = None,
        config_file_name: str = None,
    ):
        self.config = ConfigParser(allow_no_value=True)
        if config_file_name:
            self.config.read(os.path.join(path_project, config_file_name))
        else:
            self.config.read(os.path.join(path_project, "obj_detection.conf"))

        if root_dir is None:
            root_dir = self.config.get(detector_type, "root_dir", fallback="output-temp")
        if yolo_model_seragam_path is None:
            yolo_model_seragam_path = os.path.join(
                path_project,
                self.config.get(
                    detector_type,
                    "yolo_model_seragam_path",
                    fallback="data-model/object-detection-yolo8/seragam-detection/v1/best.pt"
                ),
            )

        self.root_dir = root_dir
        self.yolo_model_seragam_path = yolo_model_seragam_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()

    def load_model(self):
        self.model_seragam = YOLO(self.yolo_model_seragam_path)
        self.model_seragam.to(self.device)

    def get_dominant_color(self, image: np.ndarray) -> str:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)
        k = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = centers[np.argmax(np.bincount(labels.flatten()))].tolist()

        # Find nearest color name
        min_distance = float("inf")
        closest_color = "unknown"
        for name, rgb in color_mapping.items():
            distance = sum((a - b) ** 2 for a, b in zip(dominant_color, rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        return closest_color

    def detect(self, image_input: Union[str, np.ndarray], conf: float = 0.4) -> List[Dict]:
        objects_per_image = []

        # Image preparation
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            results = self.model_seragam(image_input, conf=conf)
        else:
            image = Image.open(image_input)
            results = self.model_seragam(image_input, conf=conf)

        torch.cuda.empty_cache()
        width, height = image.size

        data_bbox = []
        data = []

        if not isinstance(results, list):
            results = [results]

        for result in results:
            for i in range(result.boxes.shape[0]):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]  # Get YOLO class name
                confidence = float(result.boxes.conf[i].item())
                bbox = result.boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(int, bbox)
                w = x2 - x1
                h = y2 - y1

                image_array = np.array(image)
                cropped_image = image_array[y1:y1 + h, x1:x1 + w]
                color = self.get_dominant_color(cropped_image)

                print(f"DETECTED CLASS NAME: {name}")  # Debug print

                data.append({
                    "labels": name,  # ✔️ Use YOLO label directly!
                    "type": "attribute",
                    "confidence": confidence,
                    "bbox": [x1, y1, w, h]
                })

                objects_per_image.append({
                    "metadata": {
                        "image_path": image_input if isinstance(image_input, str) else "from_array",
                        "image_size": [width, height],
                        "bbox_raw": data_bbox
                    },
                    "data": data
                })

        torch.cuda.empty_cache()
        return objects_per_image

    def map_clean_result(self, raw_result: List[Dict]) -> Dict:
        try:
            attributes = [d for d in raw_result[0]["data"] if d["type"] == "attribute"]
            return {"cv_attribute": attributes}
        except Exception as e:
            print(f"Error mapping result: {e}")
            return {"cv_attribute": []}

    def classify_path(self, img_path: str) -> List[Dict]:
        return self.detect(img_path)

    def classify_url(self, url: str) -> List[Dict]:
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-agent", "Mozilla/5.0")]
            urllib.request.install_opener(opener)
            resp = urllib.request.urlopen(url)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            return self.detect(img)
        except Exception as e:
            print(f"Error reading image from URL: {e}")
            return {"cv_attribute": []}

    def classify_base64(self, base_string: str) -> List[Dict]:
        try:
            if "," in base_string:
                base_string = base_string.split(",")[1]
            decoded_data = base64.b64decode(base_string)
            np_data = np.frombuffer(decoded_data, np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            return self.detect(img)
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return {"cv_attribute": []}

    def classify_array(self, img_array: np.ndarray) -> List[Dict]:
        try:
            return self.detect(img_array)
        except Exception as e:
            print(f"Error processing image array: {e}")
            return {"cv_attribute": []}


# CLI test interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="image URL input")
    parser.add_argument("--path", help="image local path input")
    args = parser.parse_args()

    model = ObjectDetectionModel()

    if args.url:
        raw_result = model.classify_url(args.url)
        print("Cleaned:", model.map_clean_result(raw_result))

    elif args.path:
        raw_result = model.classify_path(args.path)
        print("Cleaned:", model.map_clean_result(raw_result))

    else:
        print("Error: use --url <image_url> or --path <image_local_path>")
