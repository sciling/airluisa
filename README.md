# AIRLUISA

Para lanzar el sistema es necesario tener los modelos descargados en la carpeta [_weights_ ](https://github.com/pmj110119/YOLOX_deepsort_tracker#zap-select-a-yolox-family-model) por defecto se utiliza _yolox_s.pth.tar_


### Uso
```
usage: yolo_video.py [--input] [--output] [--use-cuda] [--stream]

argumentos necesarios:
  --input        Video input path
  --output       Video output path

otros:
  --use-cuda       especificar si se quiere usar la GPU o no, en caso de tener acceso a ella.
  --stream         pasar el link de un video streaming de youtube, primero lo guardará y luego lo procesará
```
---

### Ejemplos
Una vez esté el modelo descargado para procesar un video se utiliza el siguiente comando:

```
    python3 yolo_video.py --input ../data/video_a_procesar.avi --output out/resultado.avi
```

El repositorio incluye un script para realizar sampling sobre videos antes de procesarlos:

```
    python3 sampling_videos.py video_path/ video_output_path/ num_frames
``` 

