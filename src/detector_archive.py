import sys, os
import torch
from configparser import ConfigParser
from pathlib import Path  # Added import for Path
from ultralytics import YOLO
from pprint import pprint
import cv2
import numpy as np
import urllib.request
import argparse
import base64
from PIL import Image

from typing import List, Dict
path_this = os.path.realpath(os.path.dirname(__file__))
path_project = os.path.realpath(os.path.join(path_this, ".."))
sys.path.append(path_this)
sys.path.append(path_project)
from src.helper_classes import classes_dict, color_mapping, attb_add

class ObjectDetectionModel:
    def __init__(
        self,
        root_dir: str = None,
        yolo_model_object_path: str = None,
        yolo_model_seragam_path: str = None,
        detector_type: str = "Detector",
        device: str = None,
        config_file_name: str = None,
    ):
        """
        Initializes the Detector object with the given parameters.

        Args:
            root_dir (str, optional): Root directory for output files. Defaults to None.
            yolo_model_object_path (str, optional): Path to the YOLO model for object detection. Defaults to None.
            yolo_model_seragam_path (str, optional): Path to the YOLO model for seragam detection. Defaults to None.
            device (str, optional): Device to use for inference. Defaults to None.
            detector_type (str, optional): Type of detector. Defaults to "Detector".
        """
        self.config = ConfigParser(allow_no_value=True)
        self.config_file_name = config_file_name
        if self.config_file_name is None:
            self.config.read(os.path.join(path_project, "obj_detection.conf"))
        else:
            self.config.read(os.path.join(path_project, self.config_file_name))
        
        if root_dir is None:
            root_dir = self.config.get(detector_type, "root_dir", fallback="output-temp")
        if yolo_model_object_path is None:
            yolo_model_object_path = os.path.join(path_project, self.config.get(
                detector_type, "yolo_model_object_path", fallback="/data-model/object-detection-yolo8/object-detection/v1/best.pt"  # Fixed typo in path
            ))
        if yolo_model_seragam_path is None:
            yolo_model_seragam_path = os.path.join(path_project, self.config.get(
                detector_type, "yolo_model_seragam_path", fallback="/data-model/object-detection-yolo8/seragam-detection/v1/best.pt"  # Fixed typo in path
            ))
            
        self.root_dir = root_dir
        self.output_dir = None
        self.yolo_model_object_path = yolo_model_object_path
        self.yolo_model_seragam_path = yolo_model_seragam_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.load_model()
    
    def load_model(self):
        """
        Loads the YOLO models for object detection and seragam detection.
        """
        self.model_object = YOLO(self.yolo_model_object_path)
        self.model_object.to(self.device)
        self.model_seragam = YOLO(self.yolo_model_seragam_path)
        self.model_seragam.to(self.device)

    def get_value(self, key):
        if key in classes_dict:
            return classes_dict[key]
        else:
            return "n/a"
    
    def get_dominant_color(self, image):
        """
        Returns the dominant color of an image.

        Args:
            image (numpy.ndarray): Image to be processed.

        Returns:
            str: Name of the dominant color.
        """
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape(-1, 3)
        k = 3  # You can adjust the number of clusters (colors) as needed
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dominant_color=dominant_color.tolist()
        min_distance = float('inf')
        closest_color = None
        for name, rgb in color_mapping.items():
            # Calculate the Euclidean distance between the two colors
            distance = sum((a - b) ** 2 for a, b in zip(dominant_color, rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        return closest_color
    
    def detect(self, path: str, conf: float = 0.4):
        objects_per_image = []

        object_results = self.model_object(path, conf=conf)
        torch.cuda.empty_cache()

        seragam_results = self.model_seragam(path, conf=conf)
        torch.cuda.empty_cache()

        image = Image.open(path)
        width, height = image.size

        for results, attb_add in zip([object_results, seragam_results], [True, False]):
            data_bbox = []
            data = []
            if not isinstance(results, list):
                results = [results]

            for result in results:
                for i in range(result.boxes.shape[0]):
                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]
                    confidence = float(result.boxes.conf[i].item())
                    bbox = result.boxes.xyxy[i].cpu().numpy()

                    x, y, w, h = map(int, bbox)
                    image_array = np.array(image)

                    cropped_image = image_array[y:y + h, x:x + w]
                    color = self.get_dominant_color(cropped_image)

                    if attb_add:
                        label_type = "object"
                    else:
                        label_type = "attribute"

                    data_bbox.append({
                        "xywh": [x, y, w, h],
                        "xyxy": bbox.tolist(),
                        "xywhn": result.boxes.xywhn[i].cpu().numpy().tolist(),
                        "xyxyn": result.boxes.xyxyn[i].cpu().numpy().tolist()
                    })

                    data.append({
                        "labels": self.get_value(name),
                        "type": label_type,
                        "confidence": confidence,
                        "bbox": [x, y, w, h],
                        "color": color
                    })

            obj_info = {
                "metadata": {
                    "image_path": path,
                    "image_size": [width, height],
                    "bbox_raw": data_bbox
                },
                "data": data
            }

            objects_per_image.append(obj_info)

        torch.cuda.empty_cache()
        return objects_per_image

    
    def map_clean_result(self, raw_result):
            """
            Maps the raw result of object detection to a clean result.

            Args:
                raw_result (list): A list of dictionaries containing the raw result of object detection.

            Returns:
                dict: A dictionary containing the clean result of object detection.
            """
            # get the data only 
            try:
                # clean_result = {
                # "cv_object": raw_result[0]["data"]
                # }   
                # iterate through the data and separate the objects and attributes
                clean_result = {}
                objects = []
                attributes = []
                for data in raw_result[0]["data"]:
                    if data["type"] == "object":
                        objects.append(data)
                    else:
                        attributes.append(data)
                    
                clean_result["cv_object"] = objects
                clean_result["cv_attribute"] = attributes
            except:
                clean_result = {
                "cv_object": [],
                "cv_attribute": []
                }
            
            return clean_result
    
    def classify_path(self, img_path):
        image_extraction = self.detect(img_path)
        return image_extraction

    def classify_url(self, url):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-agent", "Mozilla/5.0 (Windows NT 5.1; rv:43.0) Gecko/20100101 Firefox/43.0")]
            urllib.request.install_opener(opener)
            resp = urllib.request.urlopen(url)
            img = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            image_extraction = self.detect(img)
        except Exception as e:
            image_extraction = {"objects": []}
            print(e)
        return image_extraction

    def classify_base64(self, base_string):
        try:
            if "," in base_string:
                base_string = base_string.split(",")[1]
            decoded_data = base64.b64decode(base_string)
            np_data = np.fromstring(decoded_data, np.uint8)
            img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
            image_extraction = self.detect(img)
        except Exception as e:
            image_extraction = {"objects": []}
            print(e)
        return image_extraction

    def classify_array(self, img_array):
        try:
            image_extraction = self.detect(img_array)
        except Exception as e:
            image_extraction = {"objects": []}
            print(e)
        return image_extraction

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="image url input")
    parser.add_argument("--path", help="image local path input")
    args = parser.parse_args()

    OD = ObjectDetectionModel()

    if args.url:
        # raw_result = OD.classify_url(args.url)
        print("clean", OD.map_clean_result(OD.classify_url(args.url)))
        
    elif args.path:
        # raw_result = OD.classify_path(args.path)
        print("clean", OD.map_clean_result(OD.classify_path(args.path)))
    else:
        print("error: use either --url <image_url> or --path <image_local_path>")





