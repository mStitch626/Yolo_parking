from collections import Counter
import cv2
import numpy as np
from _lib_yolov8 import yolov8,yolov8_OCR_plate
from _lib_my import FpsCalculator,my_Rect,TrackerKalmanPlus

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TRACKER_DIST = 70
TRACKER_SKIP = 10

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TXT_video:
    def __init__(self):
        self.ids_plates = {}

    def add_key_value(self,key, value):
        if key in self.ids_plates:
            self.ids_plates[key].append(value)
        else:
            self.ids_plates[key] = [value]

    def get_Most_frequent_exiting(self,id):
        if id in self.ids_plates:
            counter = Counter(self.ids_plates[id])
            most_common_text = counter.most_common(1)[0][0] 
            del self.ids_plates[id] 
            if len(most_common_text)>0:
                return most_common_text
            else:
                return None
        else:
            return None

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_class(_objects,_class):
    list_classes = []
    for o in _objects:
        if o[5] == _class:
            list_classes.append(o[0:4])
    return list_classes

# --------------------------------------------------------------------------------
def area_test(p_area ,x_test,y_test):
    if p_area is not None and len(p_area) >= 3:
        results = cv2.pointPolygonTest(np.array(p_area,np.int32) ,((x_test,y_test)) , False )
        return  results > 0
    else:
        return False
    
# --------------------------------------------------------------------------------
def area_drow(pImage,_areas):
    if _areas is not None and len(_areas) >= 1:
        for area in _areas:
            if area is not None and len(area) >= 3:
                cv2.polylines(pImage,[np.array(area,np.int32)],True,(255,255,255),2)

# --------------------------------------------------------------------------------
def drow_objects(image:np.ndarray,objects):
    for o in objects:
        x1,y1,x2,y2,p,id_ = o
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Tracker:
    def init_area(self,polygon_points):
        print("init_area")
        areas = my_Rect.area_get(polygon_points)
        if areas is not None:
            self.area_in = areas[0]
            self.area_m = areas[1]
            self.area_out = areas[2]
        else:
            print("ERROR polygon_points is not 4 point")
            exit()
            
    def __init__(self,path_model_detector,path_model_ocr,polygon_points):
        self.init_area(polygon_points)

        self.entering1 = {}
        self.exiting1 = {}

        self.count_in = 0
        self.count_out = 0

        self.tracker = TrackerKalmanPlus(TRACKER_DIST,TRACKER_SKIP)

        self.fps_calculator = FpsCalculator()
        self.txt_vodeo = TXT_video()
        self.yolov8_car_plate = yolov8(path_model_detector ,320 , 0.5 )
        self.yolov8_ocr = yolov8_OCR_plate(path_model_ocr)

    def external_function_in(self,counter,value):
        print("external_function_in :",value)

    def external_function_out(self,counter,value):
        print("external_function_out :",value)
        
    def update(self,_image):
        objects =  self.yolov8_car_plate.process(_image)
        # 
        cars_all = get_class(objects,0)
        plates_all = get_class(objects,1)
        r_list = self.tracker.update(cars_all)
        for bbox_id in r_list:
            _x1,_y1,_x2,_y2,id = bbox_id
            x_tr,y_tr = (_x1+_x2)//2 , _y2

            plate = my_Rect.inRectMasterGetSlave([_x1,_y1,_x2,_y2],plates_all)
            # print(car,plate)
            x_tr,y_tr = (_x1+_x2)//2 , _y2
            
            if plate is not None:
                # print(car,plate)
                x1,y1,x2,y2 = int(plate[0]),int(plate[1]),int(plate[2]),int(plate[3])
                image_plate = np.array(_image[y1:y2, x1:x2])
                # cv2.imshow(f"plate ----->>>>>> ",image_plate)
                plate_txt = self.yolov8_ocr.read_plate(image_plate)
                # print(plate_txt)  
                if area_test(self.area_m ,x_tr,y_tr):
                    self.txt_vodeo.add_key_value(id,plate_txt)
                cv2.putText(_image, plate_txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
            # ++++++++++++++++++++++++++++++++++++++++++++++++++
            if area_test(self.area_in ,x_tr,y_tr):
                self.entering1[id] = ""

            if id in self.entering1:
                if area_test(self.area_out ,x_tr,y_tr):
                    del self.entering1[id]
                    self.count_in = self.count_in +  1 

                    txt = self.txt_vodeo.get_Most_frequent_exiting(id)
                    self.external_function_in( self.count_in ,txt)
                    # if txt is not None:
                    #     self.external_function_in(txt)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++
            if area_test(self.area_out ,x_tr,y_tr):
                self.exiting1[id] = ""

            if id in self.exiting1:
                if area_test(self.area_in ,x_tr,y_tr):
                    del self.exiting1[id]
                    self.count_out = self.count_out + 1 

                    txt = self.txt_vodeo.get_Most_frequent_exiting(id)
                    self.external_function_out(self.count_out,txt)
                    # if txt is not None:
                    #     self.external_function_out(txt)

            # --------------------------------------------------
            cv2.circle(_image,(x_tr,y_tr),5,(0,0,255),5)
            cv2.putText(_image,f"ID : {id}",(x_tr,y_tr-20),cv2.FONT_HERSHEY_COMPLEX,(0.75),(0,0,255),2)

        # --------------------------------------------------
        area_drow(_image,[self.area_in,self.area_m,self.area_out])
        # --------------------------------------------------
        drow_objects(_image,objects)
        # --------------------------------------------------
        self.fps_calculator.update()    
        fps = self.fps_calculator.get()
        cv2.putText(_image,'FPS : '+str(fps),(20,30),cv2.FONT_HERSHEY_COMPLEX,(0.75),(0,0,255),2)
        # --------------------------------------------------