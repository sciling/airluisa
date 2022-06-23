import sys
from timeit import default_timer as timer

import pandas as pd
import torch

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from detector import Detector
from utils.visualize import vis_detect_track, vis_df
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES

sys.path.insert(0, "./YOLOX")

class_names = COCO_CLASSES


class Tracker:
    def __init__(
        self,
        filter_class=None,
        model="yolox-s",
        ckpt="weights/yolox_s.pth.tar",
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
        start = timer()
        info = self.detector.detect(image, visual=False)
        outputs = []
        data = {}

        if info["box_nums"] > 0:
            bbox_xywh = []
            scores = []
            classes = []
            # bbox_xywh = torch.zeros((info['box_nums'], 4))
            for (x1, y1, x2, y2), class_id, score in zip(
                info["boxes"], info["class_ids"], info["scores"]
            ):
                if (
                    self.filter_class
                    and class_names[int(class_id)] not in self.filter_class
                ):
                    continue
                bbox_xywh.append(
                    [int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1]
                )
                scores.append(score)
                classes.append(class_id)

            if bbox_xywh == []:
                outputs = []
            else:
                bbox_xywh = torch.Tensor(bbox_xywh)
                # print("creo tensor")
                outputs = self.deepsort.update(bbox_xywh, scores, image)
                # print("update deepsort")
            # image = vis_track(image, outputs)

            # image = 0
            # if self.return_output_image:
            image_out = vis_detect_track(
                image, outputs, scores, classes, 0.5, class_names
            )

            df_image = vis_df(
                image_out, outputs, scores, classes, n_frame, 0.5, class_names
            )

            data["boxes"] = bbox_xywh
            data["class_ids"] = classes
            data["scores"] = scores
            data["box_nums"] = len(data["boxes"])

        else:
            df_image = pd.DataFrame()
            df_image["id_frame"] = []
            df_image["labels"] = []
            df_image["scores"] = []
            df_image["id_track"] = []

        end_track = timer()
        time_track = end_track - start
        # print("devuelvo datos")
        return image_out, outputs, data, time_track, df_image
