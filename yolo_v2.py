# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import datetime
from cv2 import threshold
import cv2
import pandas as pd

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os


import utils_detec
import psutil
import monitor_functions  

import GPUtil as GPU

vehicles = ["car", "motorbike", "bus", "truck"]
thres = 0.5

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def detect_image(self, image, n_frame):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        print("image_data", image_data)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print(out_classes)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        labels = []
        scores = []
        centers = []
        id_frame = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            #center = ((left+right)/2, (top+bottom)/2)
            center = [int((left+right)/2),int((top+bottom)/2)]
            #print(center, type(center))

            if score > thres and predicted_class in vehicles:
                id_frame.append(n_frame)
                labels.append(predicted_class)
                scores.append(score)
                centers.append(center)
                

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print("Time for frame: ", end - start)
        df = pd.DataFrame()
        df['id_frame'] = id_frame
        df['labels'] = labels
        df['center'] = centers
        df['scores'] = scores

        return image, end - start, df

    def close_session(self):
        self.sess.close()

    def init_vehicles_tracker(self, info, n_frame):

        out_boxes = info['boxes']
        out_classes = info['class_ids']
        out_scores = info['scores']

        print('Found {} boxes for {}'.format(info['box_nums'], 'img'))

        labels = []
        scores = []
        id_frame = []
        ids_tracks = []
        for i, c in list(enumerate(out_classes)):
            c = int(c)
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            if score > thres and predicted_class in vehicles:
                id_frame.append(n_frame)
                labels.append(predicted_class)
                scores.append(score)
                ids_tracks.append(0)


        df = pd.DataFrame()
        df['id_frame'] = id_frame
        df['labels'] = labels
        df['scores'] = scores
        df['id_track'] = ids_tracks

        return df


from tracker import Tracker
from detector import Detector
import imutils

def get_cpu_usage_pct():
    """
    Obtains the system's average CPU load as measured over a period of 500 milliseconds.
    :returns: System CPU load as a percentage.
    :rtype: float
    """
    return psutil.cpu_percent(interval=0.5)

def get_gpu_usage():
    GPUs = GPU.getGPUs()

    for gpu in GPUs:
        print(gpu.id, gpu.load*100, gpu.memoryUtil*100)
        deviceIds = gpu.id
        gpuUtil = gpu.load*100
        memUtil = gpu.memoryUtil*100
    
    GPU.showUtilization()
    
    return deviceIds, gpuUtil, memUtil

def detect_video(yolo, video_path, output_path=""):

    name = video_path.split("/")[-1].split(".")
    name = name[0]
    print(video_path)
    vid = cv2.VideoCapture(video_path)

    print(vid.isOpened())
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/video_fps

    print('fps = ' + str(video_fps))
    print('number of frames = ' + str(frame_count))
    print('duration (S) = ' + str(duration))
    res2 = datetime.timedelta(seconds=duration)
    print(res2)
    
    # minutes = int(duration/60)
    # seconds = duration%60
    # print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, fourcc, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    n_frame = 0
    total_frames = []
    total_time_frames = []
    init_time = timer()

    #cut number of FPS
    target = 10
    counter = 0 

    #To control the number of vehicles
    num_car = 0
    num_bike = 0
    num_bus = 0
    num_truck = 0

    num_car_per_frame = 0
    num_bike_per_frame = 0
    num_bus_per_frame = 0
    num_truck_per_frame = 0
    num_parked_per_frame = 0

    cars_frame = []
    bikes_frame = []
    bus_frame = []
    truck_frame = []

    cpu_usage = []
    cpu_freq = []
    ram_usage = []

    #deviceIds, gpuUtil, memUtil
    gpu_utils = []
    gpu_memutils = []

    #changed model yolox_s to yolox_darknet
    tracker = Tracker(filter_class=['car','truck','motorbike','bus'])
    #,model="yolov3", 
    #ckpt="weights/yolox_darknet53.47.3.pth.tar")

    # info per frame into a dataframe
    df_info_frame = pd.DataFrame()
    
    start = timer()
    while (vid.isOpened()):
        # try:
        # if counter == target: 
        return_value, frame = vid.read()
        n_frame = n_frame + 1
        total_frames.append(name+"_"+str(n_frame))
        image2, output, info, time_frame, df_per_frame = tracker.update(frame, n_frame)

        if n_frame == 1:
            df_per_frame = yolo.init_vehicles_tracker(info, n_frame)

        print(df_per_frame)
        df_info_frame = pd.concat([df_info_frame, df_per_frame])
        cv2.imshow('demo', image2)

        total_time_frames.append(time_frame)

        #cv2.imwrite(output_path+"frame"+str(n_frame)+".jpeg",image2)

        #Cuenta acumulativa
        num_car, num_bike, num_bus, num_truck = utils_detec.count_vehicles(df_per_frame, num_car, num_bike, num_bus, num_truck)

        #Por frame
        num_car_per_frame, num_bike_per_frame, num_bus_per_frame, num_truck_per_frame = utils_detec.count_vehicles(df_per_frame, num_car_per_frame, num_bike_per_frame, num_bus_per_frame, num_truck_per_frame)
        cars_frame.append(num_car_per_frame)
        bikes_frame.append(num_bike_per_frame)
        bus_frame.append(num_bus_per_frame)
        truck_frame.append(num_truck_per_frame)

        # Monitor functions for CPU and RAM
        cpu_u, cpu_f, ram_u = monitor_functions.get_results()
        cpu_usage.append(cpu_u)
        cpu_freq.append(cpu_f)
        ram_usage.append(ram_u)

        # Monitor GPU
        deviceIds, gpuUtil, memUtil = get_gpu_usage()
        gpu_utils.append(gpuUtil)
        gpu_memutils.append(memUtil)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        if isOutput:
            out.write(image2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter = 0
        num_car_per_frame = 0
        num_bike_per_frame = 0
        num_bus_per_frame = 0
        num_truck_per_frame = 0

        # else:
        #     return_value = vid.grab()
        #     counter +=1
        # except:
        #     now_time = timer()
        #     total_time = now_time - init_time
        #     res = datetime.timedelta(seconds=total_time)
        #     print("Total time: ", total_time, res)
        #     break
    sum = 0
    for i in total_time_frames:
        sum = sum + i
    print(sum, datetime.timedelta(seconds=sum))

    end_track = timer()
    time_track = end_track - start
    total_time_process  =  datetime.timedelta(seconds=time_track)

    #delete the last frame because is empty
    total_frames = total_frames[:-1]
    print("sin borrar nada: ", num_car, num_bike, num_bus, num_truck)
    num_car, num_bike, num_bus, num_truck, dict_count = utils_detec.count_vehicles_moving(df_info_frame, output_path)
    print("borrando aparcados:", num_car, num_bike, num_bus, num_truck)
    #make json with tags and metrics
    report_dict, report_percentage = utils_detec.build_results(total_frames, total_time_frames, cars_frame, bus_frame, truck_frame, bikes_frame, 
                num_car, num_bike, num_bus, num_truck)

    avg_cpu_usage, avg_cpu_freq, avg_ram_usage = utils_detec.build_monitor_cpu_results(cpu_usage, cpu_freq, ram_usage)
    print("Average use of CPU: ", avg_cpu_usage, " %")
    print("Average frequency use of CPU: ", avg_cpu_freq, " MHz")
    print("Average memory RAM: ", avg_ram_usage, " MB")
    print('RAM total is {} MB'.format(int(monitor_functions.get_ram_total() / 1024 / 1024)))

    avg_gpu_util, avg_gpu_mem = utils_detec.build_monitor_gpu_results(gpu_utils, gpu_memutils)
    print("Average use of GPU util: ", avg_gpu_util, " %")
    print("Average use of GPU mem util: ", avg_gpu_mem, " %")

    if isOutput:
        out = output_path.split(".")[0]
        utils_detec.save_json(out+".json", report_dict)
        utils_detec.save_json(out+"_percentage.json", report_percentage)
        utils_detec.save_json(out+"_id_tracks_type.json", dict_count)

        res2 = datetime.timedelta(seconds=duration)

        with open(out+"_resumen.txt", "a") as f:
            f.write("------------------------------------------\n")
            f.write('FPS = ' + str(video_fps) +"\n")
            f.write("Number of frames = " + str(frame_count) +"\n")
            f.write("Duration (S) = "+ str(duration) + ", " + str(res2)+ "\n")
            f.write("TOTAL DETECTION TIME: "+ str(datetime.timedelta(seconds=sum)) +"\n")
            f.write("TOTAL PROCES TIME:: "+ str(total_time_process) +"\n")
            f.write("------------------------------------------\n")
            f.write("Average use of CPU: "+ str(avg_cpu_usage) + " %\n")
            f.write("Average frequency use of CPU: " + str(avg_cpu_freq) + " MHz\n")
            f.write("Average memory RAM: "+ str(avg_ram_usage) + " MB\n")
            f.write("RAM total is "+ str(int(monitor_functions.get_ram_total() / 1024 / 1024)) +" MB")
            f.write("------------------------------------------\n")
            f.write("Average use of GPU util "+ str(avg_gpu_util) + " %\n")
            f.write("Average use of GPU mem util: "+ str(avg_gpu_mem) + " %\n")

    
    df_info_frame.to_csv("out/df_info_frame.csv", index=False)
    #print(report_dict)

    yolo.close_session()

def detect_dir_frames(yolo, dir_path, output_path=""):
    import glob
    import cv2

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"

    prev_time = timer()
    n_frame = 0
    total_frames = []
    total_time_frames = []
    init_time = timer()

    #cut number of FPS
    target = 1
    counter = 0 

    #To control the number of vehicles
    num_car = 0
    num_bike = 0
    num_bus = 0
    num_truck = 0

    num_car_per_frame = 0
    num_bike_per_frame = 0
    num_bus_per_frame = 0
    num_truck_per_frame = 0

    cars_frame = []
    bikes_frame = []
    bus_frame = []
    truck_frame = []

    isOutput = True if output_path != "" else False

    tracker = Tracker(filter_class=['car','truck','bike','bus'])

    for frame in sorted(glob.glob(dir_path+"*.png")):
        print(frame)
        name = frame.split("/")[-1].split(".")
        name = name[0]

        n_frame = n_frame + 1
        total_frames.append(name+"_"+str(n_frame))
        # image = Image.open(frame)
        # #image = Image.fromarray(image)
        # image, time_frame, df_per_frame = yolo.detect_image(image, n_frame)
        #df_info_frame = pd.concat([df_info_frame, df_per_frame])

        image2, output, info, time_frame, df_per_frame = tracker.update(frame, n_frame)

        if n_frame == 1:
            df_per_frame = yolo.init_vehicles_tracker(info, n_frame)

        print(df_per_frame)
        df_info_frame = pd.concat([df_info_frame, df_per_frame])
        cv2.imshow('demo', image2)

        total_time_frames.append(time_frame)

        #cv2.imwrite(output_path+"frame"+str(n_frame)+".jpeg",image2)

        #Cuenta acumulativa
        num_car, num_bike, num_bus, num_truck = utils_detec.count_vehicles(df_per_frame, num_car, num_bike, num_bus, num_truck)

        #Por frame
        num_car_per_frame, num_bike_per_frame, num_bus_per_frame, num_truck_per_frame = utils_detec.count_vehicles(df_per_frame, num_car_per_frame, num_bike_per_frame, num_bus_per_frame, num_truck_per_frame)
        cars_frame.append(num_car_per_frame)
        bikes_frame.append(num_bike_per_frame)
        bus_frame.append(num_bus_per_frame)
        truck_frame.append(num_truck_per_frame)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter = 0
        num_car_per_frame = 0
        num_bike_per_frame = 0
        num_bus_per_frame = 0
        num_truck_per_frame = 0

    now_time = timer()
    total_time = now_time - init_time
    res = datetime.timedelta(seconds=total_time)
    print("Total time: ", total_time, res)

    sum = 0
    for i in total_time_frames:
        sum = sum + i
    print(sum, datetime.timedelta(seconds=sum))

    #delete the last frame because is empty
    total_frames = total_frames[:-1]
    print("sin borrar nada: ", num_car, num_bike, num_bus, num_truck)
    num_car, num_bike, num_bus, num_truck, dict_count = utils_detec.count_vehicles_moving(df_info_frame)
    print("borrando aparcados:", num_car, num_bike, num_bus, num_truck)
    #make json with tags and metrics
    report_dict, report_percentage = utils_detec.build_results(total_frames, total_time_frames, cars_frame, bus_frame, truck_frame, bikes_frame, 
                num_car, num_bike, num_bus, num_truck)
    if isOutput:
        out = output_path.split(".")[0]
        utils_detec.save_json(out+".json", report_dict)
        utils_detec.save_json(out+"_percentage.json", report_percentage)
        utils_detec.save_json(out+"_id_tracks_type.json", dict_count)
    
    df_info_frame.to_csv("out/df_info_frame.csv", index=False)
    #print(report_dict)

    yolo.close_session()