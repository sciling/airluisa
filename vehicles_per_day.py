import json
import glob
import pandas as pd
from datetime import datetime
import sys
import argparse

#/mnt/data/videos_maquinaLocal/procesado/*
#Cameras names: 10206  10305  11505  11603  11705  1202  13702  2502  3604

#video_14_12_2022_00_24 --> día, mes, año, hora, minuto
#video_14_12_2022_00_24_id_tracks_type_0.json --> total_moving_vehicles

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
    df['date'] = dates
    df['time'] = times
    df['video_path'] = full_dirs

    print(df)
    print(df.date.min(), df.date.max())

    return df

def data_per_day(data, path_json,camera="All"):

    df = pd.DataFrame()

    days = []
    vehicles_per_day = []
    cars_per_day = []
    motorcyle_per_day = []
    buses_per_day = []
    trucks_per_day = []

    cars_per_day_perc = []
    motorcyle_per_day_perc = []
    buses_per_day_perc = []
    trucks_per_day_perc = []

    dates = list(data['date'].unique())

    for i,j in enumerate(dates):
        # print(i, j)
        date_df = data[data['date'] == dates[i]]
        list_dir = list(date_df['video_path'])

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
        
        aux_per_cars = (total_cars_per_day*100)/total_vehicles_per_day
        aux_per_motorcycles = (total_motorcyle_per_day*100)/total_vehicles_per_day
        aux_per_buses = (total_buses_per_day*100)/total_vehicles_per_day
        aux_per_trucks = (total_trucks_per_day*100)/total_vehicles_per_day
        
        days.append(j)
        vehicles_per_day.append(total_vehicles_per_day)
        cars_per_day.append(total_cars_per_day)
        motorcyle_per_day.append(total_motorcyle_per_day)
        buses_per_day.append(total_buses_per_day)
        trucks_per_day.append(total_trucks_per_day)

        cars_per_day_perc.append(round(aux_per_cars,2))
        motorcyle_per_day_perc.append(round(aux_per_motorcycles,2))
        buses_per_day_perc.append(round(aux_per_buses,2))
        trucks_per_day_perc.append(round(aux_per_trucks,2))
    
    df['date'] = days
    df['total_vehicles'] = vehicles_per_day
    df['total_cars'] = cars_per_day
    df['cars %'] = cars_per_day_perc
    df['total_motorcycle'] = motorcyle_per_day
    df['motorcycles %'] = motorcyle_per_day_perc
    df['total_buses'] = buses_per_day
    df['buses %'] = buses_per_day_perc
    df['total_trucks'] = trucks_per_day
    df['trucks %'] = trucks_per_day_perc
    df = df.sort_values(by=['date'])
    print(df)
    df.to_excel(path_json+"output.xlsx", index=False)  




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
        "--dir", nargs="?", type=str, default="/home/tgonzalez/Desktop/projects/airLUISA/videos_procesados/", help="Directory with processed videos"
    )
    parser.add_argument(
        "--path_json", nargs="?", type=str, default="/home/tgonzalez/Desktop/projects/airLUISA/results_vid_prod/", help="Directory for save results"
    )
    parser.add_argument(
        "--date", nargs="?", 
        type=str, 
        required = False,
        default="", 
        help="Date of the videos to count vehicles"
    )
    
    FLAGS = parser.parse_args()

    print(FLAGS.dir, "DIR")
    print(FLAGS.date, "Selected date")

    df = organize_data(FLAGS.dir)
    data_per_day(df, FLAGS.path_json)
