# Image Classification
The engine is used to detect object from image 

## How to use
CLI Command (image from path)
```
python detector.py --path *path/to/image/want/to/do/inference(str)* --gpu *index_of_gpu
```

CLI Command (image from path)
```
python detector.py --url *url/to/image/want/to/do/inference(str)* --gpu *index_of_gpu
```

```
Example Result
```dictionary
{
    'metadata': {
        'image _path': '/home/ronny/Yolov8/1603780301.jpg',
        'image_size': [960, 960],
        'bbox_raw': [
            {'xywh': [0, 381, 526, 578],'xyxy': [0, 381, 526, 959],'xywhn': [0, 0, 0, 0],'xyxyn': [0, 0, 0, 0]},
            {'xywh': [197, 203, 251, 65],'xyxy': [197, 203, 448, 268],'xywhn': [0, 0, 0, 0],'xyxyn': [0, 0, 0, 0]},
            {'xywh': [682, 564, 28, 36],'xyxy': [682, 564, 710, 600],'xywhn': [0, 0, 0, 0],'xyxyn': [0, 0, 0, 0]},
            {'xywh': [419, 652, 540, 307],'xyxy': [419, 652, 959, 959],'xywhn': [0, 0, 0, 0],'xyxyn': [0, 0, 0, 0]},
            {'xywh': [0, 7, 960, 953],'xyxy': [419, 652, 959, 959],'xywhn': [0, 0, 1, 960],'xyxyn': [0, 7, 960, 960]}
            ]
    },
    'data': [
        {'labels': 'class_104','type': 'atribut','confidence': 0.8865787982940674,'bbox': [0, 381, 526, 578]},
        {'labels': 'class_101','type': 'atribut','confidence': 0.7808560729026794,'bbox': [197, 203, 251, 65]},
        {'labels': 'class_123','type': 'atribut','confidence': 0.6302341222763062,'bbox': [682, 564, 28, 36]},
        {'labels': 'class_113','type': 'atribut','confidence': 0.47048404812812805,'bbox': [419, 652, 540, 307]},
        {'labels': 'Baju Karnaval','type': 'atribut','confidence': 0.727935791015625,'bbox': [0, 7, 960, 953]}
     ]
}
```

