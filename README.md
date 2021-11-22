Check clickup for documentation and sprints

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```
For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--dir] [--stream][--input] [--output]

positional arguments:
  --dir          Directory with frames to detect
  --stream       Streaming Video link of Youtube
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode
```
---

### Examples

Example for track and detect vehicles with an video input and saving results

```
python yolo_video.py --input VideoExample.mp4  --output out/example.avi

```

Expected results, jsons with info (vehicles per frame, tracking vehicles info, total percentage of vehicles per frame) and a txt with this kind of information:

```
TOTAL MOVING VEHICLES:  6
------------------------------------------
Total cars:  6
Total motorbikes:  0
Total buses:  0
Total trucks:  0
TOTAL PARKED VEHICLES:  2
------------------------------------------
Total cars:  2
Total motorbikes:  0
Total buses:  0
Total trucks:  0

PERCENTAGE VEHICLES PER FRAME:
{'total_vehicles': 768, 'type': 
[{'cars': 749, 'percentage': 0.98}, 
{'bikes': 0, 'percentage': 0.0}, 
{'buses': 0, 'percentage': 0.0}, 
{'trucks': 19, 'percentage': 0.02}]}

```






