from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import pandas as pd
import os
from glob import glob
from timeit import default_timer as timer
import datetime


import utils_detec
import monitor_functions
from monitor_gpu import Monitor
from monitor_cpu import Monitor_CPU

vehicles = ["car", "motorbike", "bus", "truck"]
thres = 0.5

def init_vehicles_tracker(info, n_frame):

        out_boxes = info['boxes']
        out_classes = info['class_ids']
        out_scores = info['scores']

        print('Found {} boxes for {}'.format(info['box_nums'], 'img'))

        labels = []
        scores = []
        id_frame = []
        ids_tracks = []
        # for i, c in list(enumerate(out_classes)):
        #     c = int(c)
        #     predicted_class = self.class_names[c]
        #     box = out_boxes[i]
        #     score = out_scores[i]

        #     if score > thres and predicted_class in vehicles:
        #         id_frame.append(n_frame)
        #         labels.append(predicted_class)
        #         scores.append(score)
        #         ids_tracks.append(0)


        df = pd.DataFrame()
        df['id_frame'] = id_frame
        df['labels'] = labels
        df['scores'] = scores
        df['id_track'] = ids_tracks

        return df

def track_cap(file):
    cap = cv2.VideoCapture(file)
    print(cap.isOpened())
    tracker = Tracker()
    a = 0
    while True:
        
        _, im = cap.read()
        if im is None:
            break
        a += 1
        if a%10!=0:
            continue
        im = imutils.resize(im, height=500)
        image,_ = tracker.update(im)
       
 
        cv2.imshow('demo', image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_video(video_path, output_path=""):
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

    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, fourcc, video_fps, video_size)

    n_frame = 0
    total_frames = []
    total_time_frames = []
    init_time = timer()

    #cut number of FPS
    target = 20
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
    # gpu_utils = []
    # gpu_memutils = []

    #changed model yolox_s to yolox_darknet
    tracker = Tracker(filter_class=['car','truck','motorbike','bus'])
    #,model="yolov3", 
    #ckpt="weights/yolox_darknet53.47.3.pth.tar")

    # info per frame into a dataframe
    df_info_frame = pd.DataFrame()
    
    start = timer()
    # monitor_gpu = Monitor(30, output_path)
    monitor_cpu = Monitor_CPU(45, output_path)
    while (vid.isOpened()):
        try:
            # if counter == target: 
            return_value, frame = vid.read()
            n_frame = n_frame + 1
            total_frames.append(name+"_"+str(n_frame))
            frame = imutils.resize(frame, height=500)
            image2, output, info, time_frame, df_per_frame = tracker.update(frame, n_frame)

            print(n_frame)

            if n_frame == 1:
                df_per_frame = init_vehicles_tracker(info, n_frame)

            # print(df_per_frame)
            df_info_frame = pd.concat([df_info_frame, df_per_frame])
            # cv2.imshow('demo', image2)

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


            #TO SAVE VIDEO
            # if isOutput:
            #     out.write(image2)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # counter = 0
            num_car_per_frame = 0
            num_bike_per_frame = 0
            num_bus_per_frame = 0
            num_truck_per_frame = 0


        # else:
        #     return_value = vid.grab()
        #     counter +=1
        except:
            now_time = timer()
            total_time = now_time - init_time
            res = datetime.timedelta(seconds=total_time)
            print("Total time: ", total_time, res)
            break
    
    sum = 0
    # monitor_gpu.stop()
    monitor_cpu.stop()
    for i in total_time_frames:
        sum = sum + i
    # print(sum, datetime.timedelta(seconds=sum))

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
            f.write("Video duration = "+ str(duration) + "s, " + str(res2)+ "\n")
            f.write("TOTAL DETECTION TIME: "+ str(datetime.timedelta(seconds=sum)) +"\n")
            f.write("TOTAL PROCES TIME:: "+ str(total_time_process) +"\n")

    
    df_info_frame.to_csv("out/df_info_frame.csv", index=False)
    #print(report_dict)
