#!/bin/bash
list_cameras="10206 10305 11505 11603 11705 1202 13702 2502 3604"
input_dir=/mnt/data/videos_maquinaLocal/salida/
output_dir=/mnt/data/videos_maquinaLocal/procesado/
#Recorre cada carpeta en la lista
for camera in $list_cameras; do
	# Entra a la carpeta
	#cd /mnt/data/videos_maquinaLocal/salida/$camera
	echo "=========="
	echo $camera
        # Recorre cada archivo en la carpeta
	for video in $input_dir$camera/*.mp4; do
           # Imprime el nombre del arhivo
	   #echo $video
	   #echo $(basename $video)
	   video_sinExtension=$(basename "$video" | sed 's/\.mp4$//')
	   #echo $video_sinExtension
	   dir_videoAvi=$input_dir$camera/$video_sinExtension.avi 
	   if test -e $dir_videoAvi
	    then echo "Archivo existe" $dir_videoAvi
            else 
		   echo "NO existe avi" $dir_videoAvi
           
                   ffmpeg -i $video -vcodec mpeg4 -acodec mp3 $dir_videoAvi
	           python yolo_video.py --input $dir_videoAvi --output $output_dir$camera/$video_sinExtension/$video_sinExtension --sampling 10 --use-cuda
            fi
	done 
	
done

