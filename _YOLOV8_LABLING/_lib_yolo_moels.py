import os
import numpy as np
from ultralytics import YOLO

class yolov8_pt:
    def __init__(self,path_model:str):
        if not os.path.exists(path_model):
            print("not exist file :",path_model)
            exit()
        self.model = YOLO(path_model)

    def process(self,image_in:np.ndarray,classes=[0],conf=0.5):
        results = self.model.predict(image_in, save=False,save_txt=False,show=False,verbose = False,conf=conf,classes=classes)
        objects = np.array(results[0].boxes.data)
        list_ = []
        for o in objects:
            x1,y1,x2,y2,p,id_ = int(o[0]), int(o[1]), int(o[2]), int(o[3]), o[4], int(o[5])
            list_.append([x1,y1,x2,y2,p,id_])
        return list_
