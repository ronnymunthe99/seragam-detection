import os,sys 
import shutil
import datetime as dt
import hashlib
import datetime as dt
import random
import argparse
import requests
from io import BytesIO

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from typing import Optional

path_this = os.path.realpath(os.path.dirname(__file__))
path_project = os.path.realpath(os.path.join(path_this, ".."))
project_root = os.path.abspath(os.path.join(path_this, "../../"))
sys.path.append(path_this)
sys.path.append(path_project)
sys.path.append(project_root)
from src.detector import ObjectDetectionModel

app = FastAPI(title=f"Object Detection API", description="Object Detection API to detect objects in an image", version="0.0.1", timeout=900)

object_detection_model = ObjectDetectionModel(device="cuda:0", config_file_name="deploy.conf")

class ObjectDetection(BaseModel):
    id: Optional[str] = None
    # image_path: Optional[str] = None
    image_url: str

def image_downloader(image_url: str) -> str:
    """
    Downloads the image from the given URL.

    Parameters
    ----------
    image_url : str
        the URL of the image to download

    Returns
    -------
    str
        the path to the downloaded image
    """
    try:
        random_name = hashlib.md5(str(random.random()).encode()).hexdigest()
        image_path = os.path.join(path_project, "temp", f"{random_name}.jpg")
        if not os.path.exists(image_path):
            # download image from url and open as PIL Image
            response = requests.get(image_url, timeout=300)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            # return as file path
            image.save(image_path)
        return image_path
    except Exception as e:
        print(e)
        return None

@app.post("/detect-object", tags=["predict"])
async def detect_object(
    id: str = Form(None),
    image_path: str = Form(None),
    image_url: str = Form(None),
    image: UploadFile = File(None),
):
    """
    This endpoint detects objects in an image.
    
    parameters:
    ----------
    id: str
        id of the image if any
    image_path: str
        path to the image
    image_url: str
        url of the image
    image: UploadFile
        image file
        
    returns:
    -------
    result: dict
        dictionary containing the result of the detection
    """
    
    if id is None:
        # hash id from random int and datetime
        id = hashlib.md5(str(random.randint(0, 1000000)).encode() + str(dt.datetime.now()).encode()).hexdigest()
        
    if image_path is not None:
        result = object_detection_model.detect(path=image_path)
        try:
            result[0]["metadata"]["id"] = id
        except:
            result = {
                "objects":"n/a"
            }
        return result
    
    if image_url is not None:
        try:
            image_path = image_downloader(image_url)
        except Exception as e:
            raise Exception(f"Failed to download image from {image_url}: {e}")
        result = object_detection_model.detect(path=image_path)
        try:
            result[0]["metadata"]["id"] = id
        except:
            result = {
                "objects":"n/a"
            }
        return result
    
    if image:
        with open(os.path.join(path_project, "temp", image.filename), "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image_path = os.path.join(path_project, "temp", image.filename)
        result = object_detection_model.detect(path=image_path)
        try:
            result[0]["metadata"]["id"] = id
        except:
            result = {
                "objects":"n/a"
            }
        return result
    
    
@app.post("/health-check", tags=["health-check"])
async def health_check(item: ObjectDetection):
    """
    This endpoint is used to check the health of the API.
    
    parameters:
    ----------
    item: ObjectDetection
        the ObjectDetection object
    
    returns:
    -------
    result: dict
        dictionary containing the result of the health check
    """
    id = item.id
    image_url = item.image_url
    
    if id is None:
        # hash id from random int and datetime
        id = hashlib.md5(str(random.randint(0, 1000000)).encode() + str(dt.datetime.now()).encode()).hexdigest()
        
    if image_url is not None:
        try:
            image_path = image_downloader(image_url)
        except Exception as e:
            raise Exception(f"Failed to download image from {image_url}: {e}")
        result = object_detection_model.detect(path=image_path)
        try:
            result[0]["metadata"]["id"] = id
        except:
            result = {
                "objects":"n/a"
            }
    return result
    
    
    