import glob
import os
import math
import random
import re
import time
import cv2
import numpy as np

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ fps
"""
class FpsCalculator:
    max_faram = 20
    def __init__(self):
        self.frame_count = 0
        self.fps = 0
        self.start_time = 0

    def update(self):
        self.frame_count += 1

        if self.frame_count <= 1:
            self.frame_count = 1
            self.start_time =  time.time()
        elif self.frame_count >= FpsCalculator.max_faram:
            self.frame_count  = 0
            end_time = time.time()
            t = end_time - self.start_time
            t_avg = t/(FpsCalculator.max_faram -1)
            self.fps = round(1/t_avg,2)

    def get(self):
        return self.fps
    
class FpsCalculator2:
    def __init__(self):
        self.fps = 0
        self.start_time = 0

    def update(self):
        if self.start_time == 0:
            self.start_time =  time.time()
        else:
            t = time.time() - self.start_time
            self.fps = 1/t
            self.start_time = 0

    def get(self):
        return self.fps
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ rect
"""
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
    # --------------------------------------------------
    def isRectangleSlaveInRectangleMaster(_rect_slave:list,_rect_master:list):
        if len(_rect_slave) >= 4 and len(_rect_master) >= 4 :
            x1,y1,x2,y2 = _rect_slave[0],_rect_slave[1],_rect_slave[2],_rect_slave[3]
            if my_Rect.isPointInRectangle((x1,y1),_rect_master) and my_Rect.isPointInRectangle((x2,y2),_rect_master) :
                return True
            else:
                return False
        else:
            return False
    # --------------------------------------------------
    def inRectMasterGetSlave(_rect_master,_list_slave):
        for s in _list_slave:
            if my_Rect.isRectangleSlaveInRectangleMaster(s,_rect_master):
                return s
        return None
    # --------------------------------------------------
    def sizeCalcule(_rect:list):
        if len(_rect) >= 4:
            x1,y1,x2,y2 = _rect[0],_rect[1],_rect[2],_rect[3]
            return abs(x2 - x1) * abs(y2 - y1)
        else:
            return 0
    # --------------------------------------------------
    def getBigRectangle(_rects:list):
        if len(_rects)  > 0 :
            index_of_max:int = 0
            for i,rect_this in enumerate(_rects):
                    size_rect_this = my_Rect.sizeCalcule(rect_this)
                    size_rect_max  = my_Rect.sizeCalcule(_rects[index_of_max])
                    if size_rect_this > size_rect_max:
                            index_of_max = i
            else:
                return _rects[index_of_max]
        else :
            return None
    # --------------------------------------------------
    def divide_straight_line(point1, point2, num_divisions):
        x1, y1 = point1
        x2, y2 = point2

        # Calculate the x and y increments for each segment
        x_increment = (x2 - x1) / num_divisions
        y_increment = (y2 - y1) / num_divisions

        # Generate the list of coordinates for the divided lines
        coordinates = []
        for i in range(num_divisions):
            x = x1 + (i * x_increment)
            y = y1 + (i * y_increment)
            coordinates.append((int(x), int(y)))

        # Add the endpoint of the original line
        coordinates.append(point2)

        return coordinates
    # --------------------------------------------------
    def divide_polygon(rectangle_points, num_divisions):
        if len(rectangle_points) != 4:
            raise ValueError("Rectangle must have exactly four points.")

        lp = my_Rect.divide_straight_line(rectangle_points[0],rectangle_points[3],num_divisions)   
        rp = my_Rect.divide_straight_line(rectangle_points[1],rectangle_points[2],num_divisions)   

        polygons = []
        for i in range(num_divisions):
            polygons.append([lp[i],rp[i],rp[i+1],lp[i+1]])
        return polygons
    # --------------------------------------------------
    def area_get(rectangle_points):
        if len(rectangle_points) != 4:
            return None

        lp = my_Rect.divide_straight_line(rectangle_points[0],rectangle_points[3],7)   
        rp = my_Rect.divide_straight_line(rectangle_points[1],rectangle_points[2],7)   
        polygons = []
        polygons.append([lp[0],rp[0],rp[2],lp[2]])
        polygons.append([lp[2],rp[2],rp[5],lp[5]])
        polygons.append([lp[5],rp[5],rp[7],lp[7]])
        return polygons
    
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ files
"""
class my_File:
    # --------------------------------------------------
    def get_all_files(directory_path:str,extension:str):
        if os.path.isdir(directory_path):
            _files = []
            for root, dirs, files in os.walk(directory_path):
                _files.extend(glob.glob(os.path.join(root, f"*.{extension}")))
            return _files
        else:
            return []

    # --------------------------------------------------
    def path_split(file_path:str):
        directory_path = os.path.dirname(file_path).replace("\\","/")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        extension = os.path.splitext(file_path)[1]  
        return [directory_path, file_name, extension]

    # --------------------------------------------------
    def find_pair_files(directory_path:str,extension1:str,extension2:str):
        files1 = my_File.get_all_files(directory_path,extension1)
        files2 = my_File.get_all_files(directory_path,extension2)

        list_pair1 = []
        list_pair2 = []

        for f1 in files1:
            for f2 in files2:
                basename_f1 = os.path.splitext(os.path.basename(f1))[0]
                basename_f2 = os.path.splitext(os.path.basename(f2))[0]
                if basename_f1 == basename_f2:
                    list_pair1.append(f1)
                    list_pair2.append(f2)
                    break
        print(f"number files pair in folder --: {directory_path} : --is-- :{len(list_pair1)} ")
        return list_pair1,list_pair2

    # --------------------------------------------------
    def rename_file(file_path:str, new_name:str):
        try:
            file_extension = os.path.splitext(file_path)[1]
            new_file_path = os.path.join(os.path.dirname(file_path), new_name + file_extension)
            os.rename(file_path, new_file_path)
        except:
            print(" <---> ERROR Rename: ",file_path)

    # --------------------------------------------------
    def concatenate_text_files(file1_path:str, file2_path:str, output_path:str):
        try:
            with open(file1_path, 'r') as file1:
                content1 = file1.read()
            with open(file2_path, 'r') as file2:
                content2 = file2.read()

            con = content1[-1]=="\n"
            if con :
                print("+++++++++++++++")
                concatenated_content = content1 + content2
                with open(output_path, 'w') as output_file:
                    output_file.write(concatenated_content)
            else:
                print("---------------")
                concatenated_content = content1 +"\n"+ content2
                with open(output_path, 'w') as output_file:
                    output_file.write(concatenated_content)
        except:
            print("error",output_path)

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ array
"""
class my_Array:
    # --------------------------------------------------
    def find_new_word_locations(array1:list, array2:str):
        new_word_locations = []
        for word in array1:
            if word in array2:
                new_word_locations.append(array2.index(word))
            else:
                new_word_locations.append(-1)
        return new_word_locations

"""'
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ image
"""
class my_Image:
    # -------------------------------------------------- 
    def convert_xyhw2xyxy(center_x:int, center_y:int, width:int, height:int):
        x1 = int(abs( center_x - width / 2))
        y1 = int(abs( center_y - height / 2))
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        return x1,y1,x2,y2
    
    # -------------------------------------------------- 
    def convert_xyxy2xywh(x1:int, y1:int, x2:int, y2:int):
        center_x = int( (x1+x2)/2)
        center_y = int( (y1+y2)/2)
        width = int(abs(x2-x1) )
        height = int(abs(y2-y1) )
        return center_x, center_y, width, height

    # -------------------------------------------------- 
    def image_place_xyxy(image_a:np.ndarray, image_b, rect:list):
        x1, y1, x2, y2 = rect

        width = abs(x2 - x1)
        height = abs(y2 - y1)
        resized_image_b = cv2.resize(image_b, (width, height))
        result = image_a.copy()
        result[y1:y2, x1:x2] = resized_image_b
        return result
    # -------------------------------------------------- 
    def image_place_xywh(image_a:np.ndarray, image_b:np.ndarray,rect:list):
        r = my_Image.convert_xyhw2xyxy(*rect)
        return  my_Image.image_place_xyxy(image_a,image_b,r)
    # -------------------------------------------------- 
    def image_place_xywh2(image_a:np.ndarray, image_b:np.ndarray,rect:list):
        imageA_w = image_a.shape[1]
        imageA_h = image_a.shape[0]
        center_x, center_y, width, height = rect
    
        if width < 1:
            aspect_ratio = image_b.shape[1] / image_b.shape[0]
            width = int(height * aspect_ratio)

        if height < 1:
            aspect_ratio = image_b.shape[0] / image_b.shape[1]
            height = int(width * aspect_ratio)

        if width < 1 and height < 1:
            width = int(image_b.shape[1])
            height = int(image_b.shape[0])
        # ----- 
        x1, y1, x2, y2 = my_Image.convert_xyhw2xyxy(center_x, center_y, width, height)
        if 0 > x1 :
            x1=0
        if x1 > imageA_w :
            x1=imageA_w -1
        if 0 > x2 :
            x2=0
        if x2 > imageA_w :
            x2=imageA_w -1

        if 0 > y1 :
            y1=0
        if y1 > imageA_h :
            y1=imageA_h -1
        if 0 > y2 :
            y2=0
        if y2 > imageA_h :
            y2=imageA_h -1
        # # ----- 
        resized_image_b = cv2.resize(image_b, (width, height))
        result = image_a.copy()
        result[y1:y2, x1:x2] = resized_image_b
        return result,[center_x, center_y, width, height ]


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Tracker
"""

class TrackerKalmanPlus:
    # dist_min test if sam object
    # Maximum consecutive frames an object can be invisible before being considered as lost
    def __init__(self,_dist_min = 200,_max_invisible_count= 10):
        self.dist_min = _dist_min
        self.max_invisible_count = _max_invisible_count

        self.center_points = {}
        self.id_count = 0
        self.kalman_filters = {}  # Store Kalman Filters for each object
        self.invisible_count = {}  # Store the number of consecutive frames an object has been invisible

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Increment the invisible count for all objects
        for object_id in self.invisible_count.keys():
            self.invisible_count[object_id] += 1

        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for object_id, pt in self.center_points.items():
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(pt))

                if dist < self.dist_min:
                    # Update Kalman Filter with new measurement
                    kalman_filter = self.kalman_filters[object_id]
                    measurement = np.array([cx, cy], dtype=np.float32).reshape(-1, 1)
                    prediction = kalman_filter.predict()
                    corrected_prediction = kalman_filter.correct(measurement)
                    corrected_state = corrected_prediction.squeeze()
                    self.center_points[object_id] = (corrected_state[0], corrected_state[1])
                    self.invisible_count[object_id] = 0  # Reset invisible count
                    objects_bbs_ids.append([x1, y1, x2, y2, object_id])
                    same_object_detected = True
                    break

            # New object is detected, assign a new ID and create a Kalman Filter for it
            if not same_object_detected:
                object_id = self.id_count
                self.center_points[object_id] = (cx, cy)
                self.invisible_count[object_id] = 0
                objects_bbs_ids.append([x1, y1, x2, y2, object_id])

                # Create Kalman Filter for the new object
                kalman_filter = cv2.KalmanFilter(4, 2)
                kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                           [0, 1, 0, 1],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]], dtype=np.float32)
                kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                            [0, 1, 0, 0]], dtype=np.float32)
                kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5
                kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
                kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)
                kalman_filter.statePost = np.array([cx, cy, 0, 0], dtype=np.float32).reshape(-1, 1)
                self.kalman_filters[object_id] = kalman_filter

                self.id_count += 1

        # Remove lost objects based on the invisible count
        lost_objects = [object_id for object_id, count in self.invisible_count.items()
                        if count > self.max_invisible_count]
        for object_id in lost_objects:
            del self.center_points[object_id]
            del self.kalman_filters[object_id]
            del self.invisible_count[object_id]

        return objects_bbs_ids

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ passed
"""
class Passed:
    def __init__(self,area1:list,area2:list):
        self.area1 =area1
        self.area2 =area2
        self.entering1 = {}
        self.entering2 = set()
        self.exiting1 = {}
        self.exiting2 = set()
        self.tracker = TrackerKalmanPlus()
    
    def area_test(p_area ,x_test,y_test):
        if p_area is not None and len(p_area) >= 3:
            results = cv2.pointPolygonTest(np.array(p_area,np.int32) ,((x_test,y_test)) , False )
            return  results > 0

    def get(self):
        return self.entering2, self.exiting2

    def update(self,_boxes_:list):
        r_list = self.tracker.update(_boxes_)
        for bbox_id in r_list:
            _x1,_y1,_x2,_y2,id = bbox_id
            x_tr,y_tr = (_x1+_x2)//2 , _y2

            if Passed.area_test(self.area1 ,x_tr,y_tr):
                self.entering1[id] = "(x_tr,y_tr)"

            if id in self.entering1:
                if Passed.area_test(self.area2 ,x_tr,y_tr):
                    self.entering2.add(id)
        
            if Passed.area_test(self.area2 ,x_tr,y_tr):
                self.exiting1[id] = "(x_tr,y_tr)"

            if id in self.exiting1:
                if Passed.area_test(self.area1 ,x_tr,y_tr):
                    self.exiting2.add(id)
        return r_list
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ocr generator_v1
"""

class generator_v1:
    def image_to_grid(img_math:np.ndarray, rectangles:list):
        for rectangle in rectangles:
            center_x, center_y, width, height = rectangle
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            cv2.rectangle(img_math, (x, y), (x + width, y + height), (0, 255, 0), 2)
        return img_math

    def get_list_random_incremental(_min_distance:int, _max_distance:int ,pMax:int):
        _last_value = 0
        _list_random = [0]
        while _last_value < (pMax):
            # print("_last_value : ",_last_value)
            _last_value = random.randint(_min_distance, _max_distance)+_last_value
            _list_random.append(_last_value)
        return _list_random

    def divide_into_grid(width:int, height:int ,p_min:int,p_max:int):
        cols = generator_v1.get_list_random_incremental(p_min,p_max,width)
        rows = generator_v1.get_list_random_incremental(p_min,p_max,height)
        lr = []
        for y in range(len(rows)-2):
            for x in range(len(cols)-2):
                cx = (cols[x]+cols[x+1])//2
                cy = (rows[y]+rows[y+1])//2
                x1= cols[x]
                x2= cols[x+1]
                y1= rows[y]
                y2= rows[y+1]
                w = x2-x1
                h = y2-y1
                # print(f"cx={cx} ,cy={cy} , w={w} , h={h} , x1={x1} , y1={y1} , x2={x2} , y2={y2}")
                lr.append([cx,cy,w,h])
                
        return lr

    def plates_gen(image_path:str,images_slave_path:list,ids_imageSlave:list,folder_out_path:str,p_min:int,p_max:int):
        image_main = cv2.imread(image_path)
        index_image_out:int = 0
        index_slave = 0
        # loop for main image
        while index_slave<len(images_slave_path):

            image_temp =  image_main.copy()
            label_image_temp = []

            imageMain_width = image_main.shape[1]-1
            imageMain_height = image_main.shape[0]-1
            rectangles = generator_v1.divide_into_grid(imageMain_width, imageMain_height,p_min,p_max)
            # image_temp = generator_v1.image_to_grid(image_temp,rectangles)
            # loop for slave image
            for rectangle in rectangles:
                if index_slave>= len(images_slave_path):
                    break
                print(index_slave)
                # print(images_slave_path)
                image_slave = cv2.imread(images_slave_path[index_slave])
                
                id_class_slave = ids_imageSlave[index_slave]

                imgSlaveC_x, imgSlaveC_y, imgSlave_w, imgSlave_h = rectangle
                # print(f"Center: ({imgSlaveC_x}, {imgSlaveC_y}), Width: {imgSlave_w}, Height: {imgSlave_h}")
                imgSlave_w = imgSlave_w-1
                imgSlave_h = imgSlave_h-1

                image_temp ,rect= my_Image.image_place_xywh(image_temp,image_slave,[imgSlaveC_x,imgSlaveC_y, imgSlave_w, imgSlave_h],imgSlave_w,imgSlave_h)
                id_class_slave =str(id_class_slave)
                cx=str(rect[0]/imageMain_width)
                cy=str(rect[1]/imageMain_height)
                w=str(rect[2]/imageMain_width)
                h=str(rect[3]/imageMain_height)

                label_slave = " ".join([id_class_slave,cx,cy,w,h])
                label_image_temp.append(label_slave)

                index_slave=index_slave+1

            # cv2.imshow('read', image_temp)
            # cv2.waitKey(1000)
            
            os.makedirs(folder_out_path, exist_ok=True)
            basename = f'{str(index_image_out).zfill(10)}_{str(random.randint(0, 999999999)).zfill(10)}'
            index_image_out = index_image_out+1
            cv2.imwrite(f"{folder_out_path}/{basename}.jpg",image_temp)

            with open(f"{folder_out_path}/{basename}.txt","w") as out:
                out.write("\n".join(label_image_temp))

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ run
"""
