import datetime
from timeit import default_timer as timer

import cv2
import numpy as np
import imutils
import pandas as pd
import streamlink

import utils_detec
from monitor_cpu import Monitor_CPU
from monitor_gpu import Monitor
from tracker import Tracker

vehicles = ["car", "motorcycle", "bus", "truck"]
thres = 0.5
saveFrames = False


def init_vehicles_tracker(info, n_frame):

    # out_boxes = info['boxes']
    # out_classes = info['class_ids']
    # out_scores = info['scores']

    print("Found {} boxes for {}".format(info["box_nums"], "img"))

    labels = []
    scores = []
    id_frame = []
    ids_tracks = []

    df = pd.DataFrame()
    df["id_frame"] = id_frame
    df["labels"] = labels
    df["scores"] = scores
    df["id_track"] = ids_tracks

    return df


def get_output(
    total_frames,
    total_time_frames,
    cars_frame,
    bus_frame,
    truck_frame,
    bikes_frame,
    output_path,
    isOutput,
    duration,
    video_fps,
    frame_count,
    df_info_frame,
    start,
    hour,
):
    end_track = timer()
    time_track = end_track - start
    total_time_process = datetime.timedelta(seconds=time_track)
    print(total_time_process)

    (
        num_car,
        num_bike,
        num_bus,
        num_truck,
        dict_count,
    ) = utils_detec.count_vehicles_moving(df_info_frame, output_path)

    sum_time_frames = sum(total_time_frames)
    # Make json with tags and metrics
    report_dict, report_percentage = utils_detec.build_results(
        total_frames,
        total_time_frames,
        cars_frame,
        bus_frame,
        truck_frame,
        bikes_frame,
        num_car,
        num_bike,
        num_bus,
        num_truck,
    )

    if isOutput:

        out = output_path.split(".")[0]
        utils_detec.save_json(out + "_" + str(hour) + ".json", report_dict)
        utils_detec.save_json(
            out + "_percentage_" + str(hour) + ".json", report_percentage
        )
        utils_detec.save_json(
            out + "_id_tracks_type_" + str(hour) + ".json", dict_count
        )

        res2 = datetime.timedelta(seconds=duration)

        with open(out + "_resumen_" + str(hour) + ".txt", "a") as f:
            f.write("------------------------------------------\n")
            f.write("FPS = " + str(video_fps) + "\n")

            if duration > 0:
                f.write("Number of frames = " + str(frame_count) + "\n")
                f.write("Video duration = " + str(duration) + "s, " + str(res2) + "\n")
                f.write(
                    "TOTAL DETECTION TIME: "
                    + str(datetime.timedelta(seconds=sum_time_frames))
                    + "\n"
                )
            f.write("TOTAL PROCES TIME:: " + str(total_time_process) + "\n")

    df_info_frame.to_csv("out/df_info_frame.csv", index=False)
    print("SAVED")

# MÉTODO AUXILIAR CONVERTIR LOS FRAMES A 640x640
def Reformat_Image(image):

    from PIL import Image
    #image = Image.open(ImageFilePath, 'r')
    #image_size = image.size
    image_size = image.shape
    width = image_size[0]
    height = image_size[1]

    if(width != height):
        #bigside = width if width > height else height
        bigside = 640

        background = Image.new('RGB', (bigside, bigside), (255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))

        background.paste(image, offset)
        #background.save('out.png')
        print("Image has been resized !")
        return background

    else:
        print("Image is already a square, it has not been resized !")
        return image


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if (len(img.shape) == 3) and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def detect_video(
    video_path, output_path="", use_cuda=True, smapling_fps=3, streaming=False
):

    if not streaming:
        name = video_path.split("/")[-1].split(".")
        name = name[0]
        print(video_path)
        vid = cv2.VideoCapture(video_path)
    else:
        name = video_path
        # pafyVid = pafy.new(video_path)
        # bestVid = pafyVid.getbest(preftype="webm")
        streams = streamlink.streams(video_path)
        vid = cv2.VideoCapture(streams["best"].url)

    # print(vid.isOpened())
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = int(vid.get(cv2.CAP_PROP_FPS))
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    num_frames_to_sample = round(video_fps / smapling_fps)

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if not streaming:
        duration = frame_count / video_fps
    else:
        duration = -1

    print(f"FPS: {video_fps}")
    print(f"Sampling FPS: {smapling_fps}")
    print(f"Number of frames to sample: {num_frames_to_sample}")
    if not streaming:
        print(f"Number of frames: {frame_count}")
        print(f"Video duration (s): {duration}")
        res2 = datetime.timedelta(seconds=duration)

    isOutput = True if output_path != "" else False

    video_output_frames_arr = []
    n_frame = 0
    total_frames = []
    total_time_frames = []
    init_time = timer()
    hour = 0

    # To control the number of vehicles
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

    is_vehicle_tracker_initialized = False

    tracker = Tracker(
        filter_class=["car", "truck", "motorcycle", "bus"],
        use_cuda=use_cuda,
        return_output_image=True if isOutput else False,
    )

    # To change model yolox_s to yolox_darknet
    # tracker = Tracker(filter_class=['car','truck','motorcycle','bus'],model="yolov3",
    # ckpt="weights/yolox_darknet53.47.3.pth.tar")

    # info per frame into a dataframe
    df_info_frame = pd.DataFrame()

    start = timer()
    last_report = start

    monitor_gpu = None
    #if use_cuda:
        #monitor_gpu = Monitor(30, output_path)  # For GPU
    #monitor_cpu = Monitor_CPU(30, output_path)  # For CPU
    while True:
        if (not streaming) and (not vid.isOpened()):
            break

        _, frame = vid.read()
        #print("FRAME SHAPE: ", frame.shape)
        try:
            image = resizeAndPad(frame, (640,640), 0)
            frame = image
            #print("FRAME RESHAPED: ", frame.shape)
        except Exception as e:
            print("EXCEPTION CATCHED: ", e)
        
        n_frame = n_frame + 1
        if (not streaming) and (n_frame >= frame_count):
            break

        if n_frame % num_frames_to_sample != 0:
            continue

        if frame is None:
            print("Error in frame: ", n_frame)
            continue
        else:
            total_frames.append(name + "_" + str(n_frame))
        
            print("Nº frame: ", n_frame)
            image2, output, info, time_frame, df_per_frame = tracker.update(frame, n_frame)

        

        if not is_vehicle_tracker_initialized:
            df_per_frame = init_vehicles_tracker(info, n_frame)
            is_vehicle_tracker_initialized = True

        df_info_frame = pd.concat([df_info_frame, df_per_frame])

        total_time_frames.append(time_frame)

        # TO SAVE Frames/VIDEO
        if isOutput:
            video_output_frames_arr.append(image2)

        # Cuenta acumulativa
        num_car, num_bike, num_bus, num_truck = utils_detec.count_vehicles(
            df_per_frame, num_car, num_bike, num_bus, num_truck
        )

        # Por frame
        (
            num_car_per_frame,
            num_bike_per_frame,
            num_bus_per_frame,
            num_truck_per_frame,
        ) = utils_detec.count_vehicles(
            df_per_frame,
            num_car_per_frame,
            num_bike_per_frame,
            num_bus_per_frame,
            num_truck_per_frame,
        )
        cars_frame.append(num_car_per_frame)
        bikes_frame.append(num_bike_per_frame)
        bus_frame.append(num_bus_per_frame)
        truck_frame.append(num_truck_per_frame)

        num_car_per_frame = 0
        num_bike_per_frame = 0
        num_bus_per_frame = 0
        num_truck_per_frame = 0

        if streaming:
            actual_time = timer()
            if (actual_time - last_report) >= 60:
                last_report = timer()
                hour += 1
                get_output(
                    total_frames,
                    total_time_frames,
                    cars_frame,
                    bus_frame,
                    truck_frame,
                    bikes_frame,
                    output_path,
                    isOutput,
                    duration,
                    video_fps,
                    frame_count,
                    df_info_frame,
                    start,
                    hour,
                )

    if isOutput:
        video_output_path = f"{output_path}_S{smapling_fps}.avi"
        height, width, _ = (
            video_output_frames_arr[0].shape
            if len(video_output_frames_arr) > 0
            else [0, 0]
        )
        size = (width, height)

        print(
            f"VIDEO SPECs: save path: {video_output_path}. FourCC: {video_FourCC}, FPS: {video_fps}, Size: {video_size}"
        )

        out = cv2.VideoWriter(
            video_output_path, cv2.VideoWriter_fourcc(*"DIVX"), video_fps, size
        )

        for i in range(len(video_output_frames_arr)):
            if saveFrames:
                cv2.imwrite(f"{output_path}_frame_{i}.jpeg", video_output_frames_arr[i])
            out.write(video_output_frames_arr[i])
        out.release()

    #if monitor_gpu is not None:
    #    monitor_gpu.stop()
    #monitor_cpu.stop()

    # Delete the last frame because is empty
    total_frames = total_frames[:-1]

    get_output(
        total_frames,
        total_time_frames,
        cars_frame,
        bus_frame,
        truck_frame,
        bikes_frame,
        output_path,
        isOutput,
        duration,
        video_fps,
        frame_count,
        df_info_frame,
        start,
        hour,
    )
