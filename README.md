# AIRLUISA

Para lanzar el sistema es necesario tener los modelos descargados en la carpeta [_weights_ ](https://github.com/pmj110119/YOLOX_deepsort_tracker#zap-select-a-yolox-family-model) por defecto se utiliza _yolox_s.pth.tar_


### Uso
```
usage: yolo_video.py [--input] [--output] [--use-cuda] [--stream] [--sampling]

argumentos necesarios:
  --input        Video input path
  --output       Video output path
  --sampling     Rate de sampling a aplicar

otros:
  --use-cuda       Especificar si se quiere usar la GPU o no, en caso de tener acceso a ella.
  --stream         Especificar si se quiere procesar un vídeo en directo o no
```
---

### Ejemplos
Una vez esté el modelo descargado para procesar un video se utiliza el siguiente comando:

```
    python3 yolo_video.py --input ../data/video_a_procesar.avi --output out/resultado.avi --sampling frames_a_samplear
```