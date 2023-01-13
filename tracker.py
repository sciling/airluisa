import sys
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import torch

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from detector import Detector
from utils.visualize import vis_detect_track, vis_df, vis_detect, vis
#from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from YOLOX.yolox.data.datasets.voc_classes import VOC_CLASSES
import cv2

sys.path.insert(0, "./YOLOX")

#class_names = COCO_CLASSES
class_names = VOC_CLASSES


class Tracker:
    def __init__(
        self,
        filter_class=None,
        model="yolox-s",
        ckpt="weights/best_ckpt.pth.tar",
        use_cuda=True,
        return_output_image=False,
    ):
        self.return_output_image = return_output_image
        self.detector = Detector(model, ckpt, use_cude=use_cuda)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=use_cuda,
        )
        self.filter_class = filter_class

    def update(self, image, n_frame):

        #print("Nº frame tracker.py: ", n_frame)

        start = timer()
        info = self.detector.detect(image, visual=False)
        outputs = []
        data = {}
        

        if info["box_nums"] > 0:
            bbox_xywh = []
            scores = []
            classes = []
    
            info_general = []
            for i,box in enumerate(info["boxes"]):
                info_general.append([list(box), info["class_ids"][i], info["scores"][i]])


            info_general_ordenado = sorted(info_general, key=lambda x: x[0][0])
           
            for (x1, y1, x2, y2), class_id, score in info_general_ordenado:
        
                if (
                    self.filter_class
                    and class_names[int(class_id)] not in self.filter_class
                ):
                    continue
                bbox_xywh.append(
                    [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
                )
                #bbox_xywh = info["boxes"] #Prueba

                scores.append(score)
                classes.append(class_id)
            
            
            if bbox_xywh == []:
                outputs = []
                classes_filtered = []
                scores_filtered = []
            else:
                bbox_xywh = torch.Tensor(bbox_xywh)
                # print("creo tensor")
                #outputs = self.deepsort.update(bbox_xywh, scores, image)

               
                outputs = self.deepsort.update(bbox_xywh, scores, image)
                #print("outputs: ", outputs)
                outputs = sorted(outputs, key=lambda x: x[0])
               
                pos_eliminar = []
                   
                iguales = False
                cont_fila_output = 0
                if len(outputs) > len(info_general_ordenado):
                    eliminarDeOutputs = True # Eliminar de outputs las detecciones extra.
                    longitud = len(info_general_ordenado)

                else:
                    eliminarDeOutputs = False
                    longitud = len(outputs)

                for i in range(longitud):
                    
                    
                    if not eliminarDeOutputs:
                        box_info_general = info_general_ordenado[i][0]
                        if iguales:
                            cont_fila_output += 1
                            box_outputs = outputs[cont_fila_output]
                        else:
                            box_outputs = outputs[cont_fila_output]
                    else:
                        
                        if iguales:
                            box_outputs = outputs[i]
                            cont_fila_output += 1
                            box_info_general = info_general_ordenado[cont_fila_output][0]
                        else:
                            box_outputs = outputs[i]
                            box_info_general = info_general_ordenado[cont_fila_output][0]

                    for j in range(4):
                        # Compruebo que la diferencia en las coordenadas de info_general y outputs sea < 5:
                        diferencia = abs(box_info_general[j] - box_outputs[j])
                        if diferencia > 5:
                            #Apunto la posicion de la fila de inof_general_ordenado a eliminar (misma posicion para score y class_id)
                            if not eliminarDeOutputs:
                                pos_eliminar.append(i)
                            else:
                                pos_eliminar.append(i)
                            #print("Eliminar de info_general: {}\n ya que no concuerda con output: {}".format(box_info_general, box_outputs))
 
                            iguales = False
                            break # Paso a la siguiente fila i, pero NO avanza la i del output -> iguales = False
                        else:
                            #print("Iguales info_general: {}\n y output: {}".format(box_info_general, box_outputs))
                            iguales = True

                #Elimino la fila eliminada en deepsort en scores y classes
                ### Los guardo en diferentes variables ya que en este caso se utiliza el "score" y "classes" originales para imprimir la imagen con solo deteccioens (sin tracking) 
                scores_filtered =  scores.copy()
                classes_filtered = classes.copy()
                if not eliminarDeOutputs:    
                    #print("SCORES ORIGINAL: ", scores_filtered)
                    scores_filtered = [score for i, score in enumerate(scores_filtered) if i not in pos_eliminar]
                    #print("SCORES FILTERED: ", scores_filtered)
                    #print("CLASSES ORIGINAL: ", classes_filtered)
                    classes_filtered = [classId for i, classId in enumerate(classes_filtered) if i not in pos_eliminar]
                    #print("CLASSES FILTERED: ", classes_filtered)
                if eliminarDeOutputs:
                    outputs = [out for i, out in enumerate(outputs) if i not in pos_eliminar]
                    #print("Outputs desp elim: ", outputs)

          
            image_out = vis_detect_track(image.copy(), outputs, scores_filtered, classes_filtered, 0.25, class_names) 
           
            df_image = vis_df(
                image_out, outputs, scores, classes, n_frame, 0.25, class_names # CAMBIO CONFG THRESHOLD
            )
           
            data["boxes"] = bbox_xywh
            data["class_ids"] = classes
            data["scores"] = scores
            data["box_nums"] = len(data["boxes"])
           

        else:

            #Auxiliar
            #image_out = 0
            image_out = vis_detect_track(
                image, outputs, [], [], 0.5, class_names
            )


            image_out = vis_detect_track(
                image, outputs, [], [], 0.5, class_names
            )


            print("info['box_nums'] = {}".format(info["box_nums"]))
            data["box_nums"] = 0
            
            df_image = pd.DataFrame()
            df_image["id_frame"] = []
            df_image["labels"] = []
            df_image["scores"] = []
            df_image["id_track"] = []

        end_track = timer()
        time_track = end_track - start
        # print("devuelvo datos")
        return image_out, outputs, data, time_track, df_image




