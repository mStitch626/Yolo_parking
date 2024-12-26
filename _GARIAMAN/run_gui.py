import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

import sys
import os
import cv2
import numpy as np

from tracker import Tracker

print(Qt)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Init data
path_video = "media/v2.mp4"
if not os.path.exists(path_video):
    print("not exist file :",path_video)
    path_video = 0
else:
    print("runing file:",path_video)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CONFIG
_poligon_points_temp = [(5, 482), (790, 477), (796, 590), (5, 586)]

# --------------------------------------------------
def area_select(point):
    global _poligon_points_temp
    if len(_poligon_points_temp) >= 4:
        print("clear ",_poligon_points_temp)
        _poligon_points_temp = []

    print("add ",point)
    _poligon_points_temp.append(point)
    if len(_poligon_points_temp) == 4:
        print("new poligon : ",_poligon_points_temp)
        tracker.init_area(_poligon_points_temp)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Tracker
path_model_detector = "models/car+plate_416.pt"
path_model_ocr = "models/ocr_224.pt"
tracker = Tracker(path_model_detector,path_model_ocr,_poligon_points_temp)

def external_function_in(counter,value):
    print(f"in  counter :  {counter}  ---  plate id : {value}")
    window.add_item_in(counter,str(value))

def external_function_out(counter,value):
    print(f"out counter :  {counter}  ---  plate id : {value}")
    window.add_item_out(counter,str(value))
    
tracker.external_function_in = external_function_in
tracker.external_function_out = external_function_out

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ GUI
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi("file.ui", self)

        self.bt_run = self.findChild(QPushButton, 'bt_run')
        self.bt_run.clicked.connect(self.bt_run_)

        self.bt_pause = self.findChild(QPushButton, 'bt_pause')
        self.bt_pause.clicked.connect(self.bt_pause_) 

        self.bt_stop = self.findChild(QPushButton, 'bt_stop')
        self.bt_stop.clicked.connect(self.bt_stop_) 

        self.l_image1 = self.findChild(QLabel, 'l_image1') 
        self.l_image1.mousePressEvent = self.label_mousePressEvent

        self.lv_model_in = QStringListModel()
        self.lv_plates_in = self.findChild(QListView, 'lv_plates_in') 
        self.lv_plates_in.setModel(self.lv_model_in)

        self.lv_model_out = QStringListModel()
        self.lv_plates_out = self.findChild(QListView, 'lv_plates_out') 
        self.lv_plates_out.setModel(self.lv_model_out)
        
        self.l_number_in = self.findChild(QLabel, 'l_number_in') 
        self.l_number_out = self.findChild(QLabel, 'l_number_out') 

        self.worker1 = WorkerScreen()
        self.worker1.ImgScreenUpdate.connect(self.ImageScreenUpdateSlot)

        self.show()
    
    def label_mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            print(f"Clicked coordinates: ({x}, {y})")
            area_select((x,y))

    def ImageScreenUpdateSlot(self, _pixmap):
        self.l_image1.setPixmap(_pixmap)

    def bt_run_(self):
        print('bt_run_')
        self.worker1.sleep = 0
        self.worker1.start()

    def bt_pause_(self):
        print('bt_pause')
        self.worker1.pauseScreen()

    def bt_stop_(self):
        print('bt_stop_')
        self.worker1.stopScreen()

    def closeEvent(self, event):
        self.worker1.stopScreen()
        event.accept()

    def add_item_in(self,number,text):
        items = self.lv_model_in.stringList()
        items.append(text)
        self.lv_model_in.setStringList(items)
        self.l_number_in.setText(f"{str(number)}")

    def add_item_out(self,number,text):
        items = self.lv_model_out.stringList()
        items.append(text)
        self.lv_model_out.setStringList(items)
        self.l_number_out.setText(f"{str(number)}")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def convert_cv_qt(cv_img): # """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(convert_to_Qt_format)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class WorkerScreen(QThread):
    ImgScreenUpdate = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.sleep = 0
        #---------------------------------------- 
        self.Capture = None
        #---------------------------------------- 
        self.ThreadActive = False

    def run(self):
        self.ThreadActive = True

        self.Capture = cv2.VideoCapture(path_video)
        while self.ThreadActive:
            time.sleep(self.sleep)
            if self.sleep != 0 :
                continue
            ret, frame = self.Capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, (800, 600))
            # --------------------------------------------------
            tracker.update(frame)
            # --------------------------------------------------
            Pic = convert_cv_qt(frame)
            self.ImgScreenUpdate.emit(Pic)
        
        self.stopScreen()

    def stopScreen(self):
        self.ThreadActive = False
        if self.Capture is not None:
            self.Capture.release()

        Pic = convert_cv_qt(np.zeros((600, 800, 3), dtype=np.uint8))
        self.ImgScreenUpdate.emit(Pic)
        cv2.destroyAllWindows()

    def pauseScreen(self):
        if self.sleep == 0:
            self.sleep = 0.5
        else:
            self.sleep = 0

app = QApplication(sys.argv)
window = MyApp()
app.exec_()