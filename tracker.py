import sys
sys.path.insert(0, './YOLOX')
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Detector
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from utils.visualize import vis_track, vis, vis_detect_track, vis_df
from timeit import default_timer as timer



class_names = COCO_CLASSES

class Tracker():
    def __init__(self, filter_class=None, model='yolox-s', ckpt='weights/yolox_s.pth.tar', ):
        self.detector = Detector(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_class = filter_class

    def update(self, image, n_frame):
        start = timer()
        info = self.detector.detect(image, visual=False)
        end_detec = timer()
        time_detec = end_detec - start
        outputs = []
        data = {}
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            classes = []
            #bbox_xywh = torch.zeros((info['box_nums'], 4))
            for (x1, y1, x2, y2), class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                    continue

                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                scores.append(score)
                classes.append(class_id)

            data['boxes'] = bbox_xywh
            data['class_ids'] = classes
            data['scores'] = scores
            data['box_nums'] = len(data['boxes'])

            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, image)
            end_track = timer()
            time_track = end_track - start

            #image = vis_track(image, outputs)
            image = vis_detect_track(image, outputs, scores, classes, 0.5, class_names)
            df_image = vis_df(image, outputs, scores, classes, n_frame, 0.5, class_names)


        print(time_detec)
        return image, outputs, data, time_detec, df_image
