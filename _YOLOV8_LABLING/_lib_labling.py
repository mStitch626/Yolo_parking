import glob
import os
import cv2
import numpy as np

# ===============================================================================================================
colors = [(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),
          (100,0,0),(0,100,0),(100,100,0),(0,0,100),(100,0,100),]
colors = colors + np.random.uniform(0, 250, size=(100, 3)).tolist()

# ===============================================================================================================
def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# ===============================================================================================================
class my_Rect:
    # --------------------------------------------------
    def _isBetweenNumber(v: int, a: int, b: int) -> bool:
        if (a <= v <= b) or (a >= v >= b):
            return True
        else:
            return False
    # --------------------------------------------------
    def isPointInRectangle(v: list, rect: list) -> bool:
        if len(rect) >= 4:
            x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
            x0, y0 = v
            if my_Rect._isBetweenNumber(x0, x1, x2) and my_Rect._isBetweenNumber(y0, y1, y2):
                return True
            else:
                return False
        else:
            return False

# ===============================================================================================================      
class my_File:
    def get_all_files(directory_path:str,extension:str):
        if os.path.isdir(directory_path):
            _files = []
            for root, dirs, files in os.walk(directory_path):
                _files.extend(glob.glob(os.path.join(root, f"*.{extension}")))
            return _files
        else:
            return []

# =============================================================================================================== 
class my_Labling:
    def read_classes(path_classes):
        if not os.path.exists(path_classes):
            list_classes = [(lambda x: f"{x}")(i) for i in range(100)]  
            my_Labling.save_classes(path_classes,list_classes)
        with open(path_classes, 'r') as r:
            return r.read().split("\n")

    def save_classes(path_classes,list_classes):
        with open(path_classes,"w") as out:
            out.write("\n".join(list_classes))
        
    def read_file_label_float(label_path:str):
        try:
            with open(label_path, 'r') as label_file:
                label_data = label_file.readlines()
            label_coordinates_per = []
            for line in label_data:
                r= line.split(" ")
                _id_class = int(r[0])
                lable_per_this = [_id_class,round(float(r[1]),3) ,round(float(r[2]),3) ,round(float(r[3]),3) ,round(float(r[4]),3) ]
                label_coordinates_per.append(lable_per_this)
            return label_coordinates_per
        
        except:
            return []
        
    def save_label_file_float(label_path:str,objects:list):
        try:
            lines = []
            for o in  objects:
                lines.append(' '.join(map(str, o)))

            with open(label_path,"w") as out:
                out.write("\n".join(lines))

        except:
            print("Error : save_file_label_float ")
        
    def convert_object_float_to_xyxy(image_width, image_height,label_line_float):
        id_, xc, yc, w, h = label_line_float
        id_, xc, yc, w, h =  [id_,int(xc*image_width), int(yc * image_height),int( w*image_width),int( h*image_height)]
        x1 = xc - ( w // 2) 
        y1 = yc - ( h // 2)
        x2 = x1 + w
        y2 = y1 + h
        return [id_,x1,y1,x2,y2]
    
    def convert_object_xyxy_to_float(image_width, image_height,idxyxy):
        id_ , x1 , y1 , x2 , y2 = idxyxy
        xc = round(((x1+x2)//2)/image_width,3)
        yc = round(((y1+y2)//2)/image_height,3)
        w = round((abs(x2-x1))/image_width,3)
        h = round((abs(y2-y1))/image_height,3)
        return [id_,xc,yc,w,h]

    def drow_objects_float(image:np.ndarray,labels_float_xywh,classes,class_index_only=None):
        index = 0
        for cord in labels_float_xywh:
            _id,_x1,_y1,_x2,_y2 = my_Labling.convert_object_float_to_xyxy(image.shape[1],image.shape[0],cord)

            color = colors[_id]
            text =f"{classes[_id]}"
            if class_index_only is not None:
                if _id == class_index_only:
                    my_Labling.drow_rect(image,_x1,_y1,_x2,_y2,color,text)
            else:
                    my_Labling.drow_rect(image,_x1,_y1,_x2,_y2,color,text)
            index +=1

    def drow_rect(_image,_x1,_y1,_x2,_y2,_color,_txet):
        rectangle_layer = np.zeros_like(_image)
        cv2.rectangle(rectangle_layer, (_x1, _y1), (_x2, _y2), _color, thickness=cv2.FILLED)
        _image[:] = cv2.addWeighted(_image, 1, rectangle_layer, 0.3, 0)
        cv2.rectangle(_image, (_x1, _y1), (_x2, _y2), _color, thickness=2)
        cv2.putText(_image, _txet , (_x1 + 5, _y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (99,99,99) , 2)


class Labeling_item:
    def __init__(self):
        self.path_label = None
        self.objects_all = []

        self.path_classes = None
        self.classes_all = []
        self.class_index_selected=None

        self.path_image:str = None
        self.image:np.ndarray = None
        self.image_w:int = None
        self.image_h:int = None

    def set_class_only(self,class_index_selected=None):
        self.class_index_selected = class_index_selected

    def set(self,path_image:str):
        # ++++++++++++++++++++++++++++++ path image
        self.path_image = path_image

        # ++++++++++++++++++++++++++++++ path label
        if self.path_image is not None: 
             self.path_label = self.path_image[:-3]+"txt"
        else:
            self.path_label = None

        # ++++++++++++++++++++++++++++++ read image
        self._read_image()

        # ++++++++++++++++++++++++++++++ read_classes
        self._read_classes()

        # ++++++++++++++++++++++++++++++ read_objects
        self._read_objects()

        # ++++++++++++++++++++++++++++++ drow_object
        self._drow_object()

        # ++++++++++++++++++++++++++++++ 

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_image(self):
        if self.path_image is not None or self.path_image != "":
            imgsz = 640 
            cv_img = cv2.imread(self.path_image)
            [height, width, _] = cv_img.shape
            length = max((height, width))
            _w = int((width / length)* imgsz)
            _h = int((height / length)* imgsz)
            self.image = cv2.resize(cv_img, (_w,_h))
            
            self.image_w = self.image.shape[1]
            self.image_h = self.image.shape[0]
        else:
            self.image = None
            self.image_w = None
            self.image_h = None
            
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_objects(self):
        if self.path_label is not None: 
            self.objects_all = my_Labling.read_file_label_float(self.path_label)
        else:
            self.objects_all = []

    def _drow_object(self):
        if self.objects_all is not None and self.classes_all is not None:
            my_Labling.drow_objects_float(self.image ,self.objects_all ,self.classes_all ,self.class_index_selected)

    def add_new_object(self,x1,y1,x2,y2,class_id):
        if self.path_label != "":    
            o = my_Labling.convert_object_xyxy_to_float(self.image_w,self.image_h,[class_id,x1,y1,x2,y2])
            self.objects_all.append(o)
            my_Labling.save_label_file_float(self.path_label,self.objects_all)
    
    def delete_object(self,object_index):
        if self.objects_all is not None and self.path_label is not None and len(self.objects_all)>0:
            object_index = clamp(object_index,0,len(self.objects_all)-1)
            del self.objects_all[object_index]
            my_Labling.save_label_file_float(self.path_label,self.objects_all)

    def delete_object_xy(self,_x,_y):
        if self.objects_all is not None and self.path_label is not None and len(self.objects_all)>0:
            for i in range(len(self.objects_all)):
                try:
                    i = clamp(i,0,len(self.objects_all))
                    o_float = self.objects_all[i]
                    _,x1,y1,x2,y2 = my_Labling.convert_object_float_to_xyxy(self.image_w,self.image_h,o_float)

                    if my_Rect.isPointInRectangle([_x,_y],[x1,y1,x2,y2]):
                        self.delete_object(i)
                except:
                    print("delete_object_xy : error index :",i)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_classes(self):
        if self.path_image != None:
            path_directory = os.path.dirname(self.path_image).replace("\\","/")
            self.path_classes = os.path.join(path_directory,"classes.txt")
            self.classes_all =  my_Labling.read_classes(self.path_classes)

    def add_new_class(self):
        if self.classes_all is not None and self.path_label is not None:
            l = len(self.classes_all)
            self.classes_all.append(f"new_class_{l}")
            my_Labling.save_classes(self.path_classes,self.classes_all)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def delete(self):
        try:
            os.remove(self.path_image)
        except:
            pass
        try:
            os.remove(self.path_label)
        except:
            pass
        print("remove")