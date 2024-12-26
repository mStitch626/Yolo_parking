import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

import sys
import cv2

from _lib_labling import Labeling_item ,my_File ,clamp
from _lib_yolo_moels import yolov8_pt
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ GUI

class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi("./_YOLOV8_LABLING/file.ui", self)

        self.item_selected = Labeling_item()
        self.files_index   = 0
        self.classes_index = 0
        self.objects_index = 0
        self.files_jpg_all = []
 
        self.object_new = [None,None,None,None]
        
        self.et_path_folder     = self.findChild(QLineEdit, 'et_path_folder')
        self.et_selected_class  = self.findChild(QLineEdit, 'et_selected_class')
        self.et_selected_files  = self.findChild(QLineEdit, 'et_selected_file')

        self.cb_class_only  = self.findChild(QCheckBox, 'class_only')

        self.bt_folder = self.findChild(QPushButton, 'bt_folder')
        self.bt_folder.clicked.connect(self.bt_folder_)

        self.bt_new_class = self.findChild(QPushButton, 'bt_new_class')
        self.bt_new_class.clicked.connect(self.bt_new_class_)

        self.bt_delete_file = self.findChild(QPushButton, 'bt_delete_file')
        self.bt_delete_file.clicked.connect(self.bt_delete_file_)

        self.bt_previous = self.findChild(QPushButton, 'bt_previous')
        self.bt_previous.clicked.connect(self.bt_previous_) 

        self.bt_next = self.findChild(QPushButton, 'bt_next')
        self.bt_next.clicked.connect(self.bt_next_) 

        self.l_image_show = self.findChild(QLabel, 'l_image1') 
        self.l_image_show.mousePressEvent = self.label_mousePressEvent

        self.init_QListView()
        self.init_model()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ model

    def  init_model(self):
        self.bt_model_path = self.findChild(QPushButton, 'bt_model_path')
        self.bt_model_path.clicked.connect(self.bt_model_path_)

        self.bt_model_annotation = self.findChild(QPushButton, 'bt_model_annotation')
        self.bt_model_annotation.clicked.connect(self.bt_model_annotation_)

        self.et_model_path     = self.findChild(QLineEdit, 'et_model_path')
        self.et_model_class     = self.findChild(QLineEdit, 'et_model_class')

        self.model = None

    def bt_model_path_(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "pt Files (*.pt)", options=options)

        if file_path:
            self.et_model_path.setText(file_path)
            self.model = yolov8_pt(file_path)

    def bt_model_annotation_(self):
        item = Labeling_item()
        for f in self.files_jpg_all:
            try:
                print(f)
                item.set(f)
                model_class = int(self.et_model_class.text())
                rs = self.model.process(item.image,[model_class],0.5)
                for r in rs :
                    print(r)
                    x1,y1,x2,y2,p,id_ = r
                    item.add_new_object(x1,y1,x2,y2,self.classes_index)
            except:
                print("error")
        QMessageBox.question(self, "OK", "END annotation", QMessageBox.Yes | QMessageBox.No)
        

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ QListView
    def init_QListView(self):
        self.lv_model_files = QStringListModel()
        self.lv_files = self.findChild(QListView, 'lv_files') 
        self.lv_files.setModel(self.lv_model_files)
        self.lv_files.selectionModel().selectionChanged.connect(self.lv_files_Changed)

        self.lv_model_classes = QStringListModel()
        self.lv_classes = self.findChild(QListView, 'lv_classes') 
        self.lv_classes.setModel(self.lv_model_classes)
        self.lv_classes.selectionModel().selectionChanged.connect(self.lv_classes_Changed)

        self.refresh()

    def lv_files_Changed(self, selected, deselected):
        self.files_index = selected.indexes()[0].row()
        index = self.lv_model_files.index(self.files_index, 0, QModelIndex())
        self.lv_files.setCurrentIndex(index)
        self.refresh()

    def lv_classes_Changed(self, selected, deselected):
        self.classes_index = selected.indexes()[0].row()
        self.refresh()

    def bt_previous_(self):
        self.files_index -= 1
        self.files_index = clamp(self.files_index,0,len(self.files_jpg_all)-1 )
        index = self.lv_model_files.index(self.files_index, 0, QModelIndex())
        self.lv_files.setCurrentIndex(index)

    def bt_next_(self):
        self.files_index += 1
        self.files_index = clamp(self.files_index,0,len(self.files_jpg_all)-1 )
        index = self.lv_model_files.index(self.files_index, 0, QModelIndex())
        self.lv_files.setCurrentIndex(index)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ refresh view
    def refresh(self):
        self.object_new = [None,None,None,None]
        self.show_img()
        self.show_list_files()
        self.show_list_classes()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EVENT mouse
    def label_mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()

            if self.object_new [0] is None:
                self.object_new[0] = x
                self.object_new[1] = y
                
            elif self.object_new[0] is not None and self.object_new[2] is None:
                self.object_new[2] = x
                self.object_new[3] = y
                x1,y1,x2,y2 = self.object_new
                self.object_new = [None,None,None,None]

                self.item_selected.add_new_object(x1,y1,x2,y2,self.classes_index)
                self.refresh()
        
        if event.button() == Qt.RightButton:
            x = event.x()
            y = event.y()

            self.item_selected.delete_object_xy(x,y)
            self.refresh()
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EVENT folder project
    def bt_folder_(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_path = dialog.getExistingDirectory()
        self.files_jpg_all = my_File.get_all_files(folder_path,"jpg")

        self.et_path_folder.setText(folder_path)

        self.files_index   = 0
        self.classes_index = 0
        self.objects_index = 0

        self.refresh()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EVENT class
    def bt_new_class_(self):
        self.item_selected.add_new_class()
        self.refresh()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EVENT file delete
    def bt_delete_file_(self):
        confirm_dialog = QMessageBox.question(self, "Confirm Deletion", f"Do you want to delete the files ?", QMessageBox.Yes | QMessageBox.No)
        if confirm_dialog == QMessageBox.Yes:
            self.item_selected.delete()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ function Image
    def show_img(self):
        try:
            path_image = self.files_jpg_all[self.files_index]
            
            if self.cb_class_only.isChecked():
                self.item_selected.set_class_only(self.classes_index)
                self.item_selected.set(path_image)
            else:
                self.item_selected.set_class_only(None)
                self.item_selected.set(path_image)

            if self.item_selected.image is not None:
                qt_img = self.convert_cv_qt(self.item_selected.image)
                self.l_image_show.resize(self.item_selected.image_w, self.item_selected.image_h)
                self.l_image_show.setPixmap(qt_img)
        except:
            self.l_image_show.resize(1,1)
    
    def convert_cv_qt(self,cv_img): # """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ functions list
    def show_list_files(self):
        try:
            # list file 
            list_files = []
            for f_index,path_file in enumerate(self.files_jpg_all):
                file_name = os.path.splitext(os.path.basename(path_file))[0]
                list_files.append(f"{f_index} : {file_name}") 
            self.lv_model_files.setStringList(list_files)
            # selected file
            if list_files is not None and len(list_files) > 0:
                self.files_index = clamp(self.files_index,0,len(self.files_jpg_all)-1)    
                file_this = list_files[self.files_index]
                self.et_selected_files.setText(file_this)
        except:
            self.lv_model_files.setStringList([])
            self.et_selected_files.setText("")

    def show_list_classes(self):
        try:
            # list class
            self.lv_model_classes.setStringList(self.item_selected.classes_all)
            # selected class
            if self.item_selected.classes_all is not None:
                self.classes_index = clamp(self.classes_index,0,len(self.item_selected.classes_all)-1)
                class_this = self.item_selected.classes_all[self.classes_index]
                self.et_selected_class.setText(class_this)
        except:
            self.lv_model_classes.setStringList([])
            self.et_selected_class.setText("")

app = QApplication(sys.argv)
window = MyApp()
window.show()
app.exec_()