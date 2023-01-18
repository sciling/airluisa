#!/bin/sh


search_dir=/mnt/data/10206/
output_dir=/mnt/data/out/10206/
correg=_Corr
#list = "video_22_07_2022_14_00.mp4 video_22_07_2022_14_00.mp4.avi video_22_07_2022_15_00.mp4.avi video_22_07_2022_15_00.mp4 video_22_07_2022_16_00.mp4"
for entry in `ls $search_dir`; do
    echo $search_dir$entry
    if [ $entry = video_22_07_2022_14_00.mp4.avi -o $entry = video_22_07_2022_15_00.mp4.avi -o $entry = video_22_07_2022_16_00.mp4.avi -o $entry = video_22_07_2022_17_00.mp4.avi -o $entry = video_22_07_2022_18_00.mp4.avi -o $entry = video_22_07_2022_19_00.mp4.avi -o $entry = video_22_07_2022_20_00.mp4.avi -o $entry = video_22_07_2022_21_00.mp4.avi -o $entry = video_22_07_2022_22_00.mp4.avi -o $entry = video_22_07_2022_23_00.mp4.avi -o $entry = video_23_07_2022_00_00.mp4.avi -o $entry = video_23_07_2022_01_00.mp4.avi -o $entry = video_23_07_2022_02_00.mp4.avi -o $entry = video_23_07_2022_03_00.mp4.avi -o $entry = video_23_07_2022_04_00.mp4.avi -o $entry = video_23_07_2022_05_00.mp4.avi -o $entry = video_23_07_2022_06_00.mp4.avi -o $entry = video_23_07_2022_07_00.mp4.avi -o $entry = video_23_07_2022_08_00.mp4.avi -o $entry = video_23_07_2022_09_00.mp4.avi -o $entry = video_23_07_2022_10_00.mp4.avi ]; 
    then
    	echo "In the list"
    else
 
    	ffmpeg -i $search_dir$entry -vf scale=2560:1440:flags=lanczos $search_dir$entry.avi
    	python yolo_video.py --input $search_dir$entry.avi --output $output_dir$entry$correg --sampling 10 --use-cuda
    
    fi 
	  
    #ffmpeg -i $search_dir$entry -vf scale=2560:1440:flags=lanczos $search_dir$entry.avi
    #python yolo_video.py --input $search_dir$entry.avi --output $output_dir$entry$correg --sampling 10 --use-cuda
		#python yolo_video.py --input $search_dir$entry.avi --output $output_dir$entry$correg --sampling 10 --use-cuda
	 
   
done

