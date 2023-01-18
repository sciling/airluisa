import json
import glob
import pandas as pd
from datetime import datetime
import sys
import argparse

def organize_data(directory):
    df = pd.DataFrame()

    full_dirs = []
    vid_names = []
    dates = []
    times = []
    cameras = []

    for file in sorted(glob.glob(directory+'*/*'),reverse=True):
        full_dir = file
        vid_name = file.split('/')[-1]
        cam_name = file.split('/')[-2]
        data_video = vid_name.split('_')
        data = '/'.join([data_video[1],data_video[2],data_video[3]])
        data_obj = datetime.strptime(data, '%d/%m/%Y')
        time = ':'.join([data_video[4],data_video[5]])
        time_obj = datetime.strptime(time,'%H:%M').time()

        full_dirs.append(full_dir)
        vid_names.append(vid_name)
        dates.append(data_obj)
        times.append(time_obj)
        cameras.append(cam_name)
        

    df['video_name'] = vid_names
    df['cam_num'] = cameras
    df['data'] = dates
    df['time'] = times
    df['video_path'] = full_dirs

    print(df)

    return df

def count_vehicles_data(list_dir, path_json, date, date2, camera="All"):

    total_vehicles_per_day = 0
    total_cars_per_day = 0
    total_motorcyle_per_day = 0
    total_buses_per_day = 0
    total_trucks_per_day = 0

    for dir in list_dir:
        for file in sorted(glob.glob(dir+'/*_id_tracks_type_0.json')):
            with open(file) as f:
                data_json = json.load(f)

            total_moving_vehicles = data_json['total_moving_vehicles']['num']
            total_moving_cars = data_json['total_moving_vehicles']['cars']
            total_moving_motorcycles = data_json['total_moving_vehicles']['motorcycles']
            total_moving_buses = data_json['total_moving_vehicles']['buses']
            total_moving_trucks = data_json['total_moving_vehicles']['trucks']

            total_vehicles_per_day = total_vehicles_per_day + total_moving_vehicles
            total_cars_per_day = total_cars_per_day + total_moving_cars
            total_motorcyle_per_day = total_motorcyle_per_day + total_moving_motorcycles
            total_buses_per_day = total_buses_per_day + total_moving_buses
            total_trucks_per_day = total_trucks_per_day + total_moving_trucks

    
    data = {}
    data['date1'] = date
    data['date2'] = date2
    data['camera'] = camera
    data['total_moving_vehicles'] = total_vehicles_per_day
    data['total_cars'] = total_cars_per_day
    data['total_motorcycles'] = total_motorcyle_per_day
    data['total_buses'] = total_buses_per_day
    data['total_trucks'] = total_trucks_per_day

    with open(path_json+'results.json', "w") as outfile:
        json.dump(data, outfile)
    
    if date2 is None:
        print('Date:', date)
    else:
        print('Date:', date, '-', date2)
    print('Camera:', camera)
    print("TOTAL MOVING VEHICLES: ", total_vehicles_per_day)
    print("------------------------------------------")
    print("Total cars: ", total_cars_per_day)
    print("Total motorcycles: ", total_motorcyle_per_day)
    print("Total buses: ", total_buses_per_day)
    print("Total trucks: ", total_trucks_per_day)

    print("Info saved at: ", path_json)

FLAGS = None

if __name__ == "__main__":
    # params to get data
    parser = argparse.ArgumentParser()
    """
    Command line options
    """
    parser.add_argument(
        "--dir", nargs="?", type=str, default="/mnt/data/videos_maquinaLocal/procesado/", help="Directory with processed videos"
    )
    parser.add_argument(
        "--path_json", nargs="?", type=str, default="/home/ihab/airLUISA/results_stats/", help="Directory for save results"
    )
    parser.add_argument(
        "--date", nargs="?", 
        type=str, 
        required = True,
        default="", 
        help="Date of the videos to count vehicles"
    )
    parser.add_argument(
        "--date2", nargs="?", 
        type=str, 
        required = False,
        default=None, 
        help="Weeks or months of the videos to count vehicles"
    )
    parser.add_argument(
        "--camera", nargs="?", 
        type=str, 
        required = False,
        default=None, 
        help="Camera of the videos to count vehicles"
    )
    
    FLAGS = parser.parse_args()

    print(FLAGS.dir, "DIR")
    print(FLAGS.date, "Selected date")

    df = organize_data(FLAGS.dir)

    if FLAGS.date2 is None:
        date_df = df[df['data'] == FLAGS.date]
    else:
        date_df = df[(df['data'] > FLAGS.date) & (df['data'] < FLAGS.date2)]
    
    if FLAGS.camera is None:
        list_dir = list(date_df['video_path'])
        # list_vid_names = list(select_df['video_name'])
        count_vehicles_data(list_dir, FLAGS.path_json, FLAGS.date, FLAGS.date2)
    else:
        select_df = date_df[date_df['cam_num'] == FLAGS.camera]
        list_dir = list(select_df['video_path'])
        count_vehicles_data(list_dir, FLAGS.path_json, FLAGS.date, FLAGS.date2, FLAGS.camera)

    # Example + info:
    # python3 vehicles_stats.py --date '14/12/2022' --date2 '19/12/2022' --camera 3604
    # video_14_12_2022_00_24 --> día, mes, año, hora. minuto
    # video_14_12_2022_00_24_id_tracks_type_0.json --> total_moving_vehicles