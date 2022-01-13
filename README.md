**AIRLUISA**

Para lanzar el sistema es necesario tener los modelos descargados en la carpeta [_weights_ ](https://github.com/pmj110119/YOLOX_deepsort_tracker#zap-select-a-yolox-family-model) por defecto se utiliza _yolox_s.pth.tar_

Una vez esté el modelo descargado para lanzar pruebas se utiliza el siguiente comando:

`python3 yolo_video.py --input ../data/video_a_procesar.avi --output out/resultado.avi`

El repositorio incluye un script para realizar sampling sobre videos antes de procesarlos:

`python3 sampling_videos.py video_path/ video_output_path/ num_frames`

