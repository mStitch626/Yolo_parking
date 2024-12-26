import os
import cv2
import numpy as np
from ultralytics import YOLO

class yolov8_OCR_plate:
    array_names_ocr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                                'T', 'U', 'V', 'W', 'X', 'Y','Z']
    def __init__(self,path_model_ocr):
        self.yolov8_ocr = yolov8( path_model_ocr , 224 , 0.5 )

    def read_plate(self,_image):
        objects_ocr = self.yolov8_ocr.process(_image)
        
        objects_ocr.sort(key=lambda x: x[0])
        list_char = []
        for o_ocr in objects_ocr:
            id_object = o_ocr[5]
            char = yolov8_OCR_plate.array_names_ocr[id_object]
            list_char.append(char)

        plate_txt =  ''.join(list_char)
        return plate_txt       
    
class yolov8:
    def __init__(self,_path_model:str , _imgsz:int , _conf:float):
        if not os.path.exists(_path_model):
            print("not exist file :",_path_model)
            exit()

        self.conf=_conf
        self.imgsz=_imgsz
        
        self.model = YOLO(_path_model)

    def process(self,image_in:np.ndarray):
        results = self.model.predict(image_in, task='detect',save=False,save_txt=False,show=False,verbose = False,imgsz= self.imgsz,conf=self.conf,device="cpu")
        objects = np.array(results[0].boxes.data)
        list_ = []
        for o in objects:
            x1,y1,x2,y2,p,id_ = int(o[0]), int(o[1]), int(o[2]), int(o[3]), o[4], int(o[5])
            list_.append([x1,y1,x2,y2,p,id_])
        return list_

    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# path_model   = "models/car+plate_416/best.pt"
# # path_model   = "models/car+plate_416/best.onnx"
# # path_model   = "models/car+plate_416/best.tflite"

# model = yolov8(path_model)
# model.imgsz = 416

# img = cv2.imread("image.jpg")
# objects = model.process(img)

# def drow_objects(image:np.ndarray,objects):
#     for o in objects:
#         x1,y1,x2,y2,p,id_ = o
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# drow_objects(img,objects)

# cv2.imshow("out",img)
# cv2.waitKey(0)