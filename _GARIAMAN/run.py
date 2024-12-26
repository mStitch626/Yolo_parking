import time
import cv2
from tracker import Tracker

_poligon_points_temp = [(5, 482), (790, 477), (796, 590), (5, 586)]
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ functions
WINDOW_NAME = "WIN"

def WIN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN :  
        print(f"area : ({x},{y}),")
        area_select((x,y))

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, WIN)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
def area_select(point):
    global _poligon_points_temp
    if len(_poligon_points_temp) >= 4:
        print("clear ",_poligon_points_temp)
        _poligon_points_temp = []

    print("add ",point)
    _poligon_points_temp.append(point)
    if len(_poligon_points_temp) == 4:
        print("new poligon : ",_poligon_points_temp)
        passed.init_area(_poligon_points_temp)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ trcker
path_model_detector = "models/car+plate_416.pt"
path_model_ocr = "models/ocr_224.pt"
# path_model_detector = "models/car+plate_416/best_ncnn_model_416/"
# path_model_detector = "models/car+plate_416/best.onnx"
# path_model_detector = "models/car+plate_416/best.tflite"
# path_model_ocr = "models/ocr_224/best.onnx"
# path_model_ocr = "models/ocr_224/best.tflite"
passed = Tracker(path_model_detector,path_model_ocr,_poligon_points_temp)

def external_function_in(counter,value):
    print(f"in  counter :  {counter}  ---  plate id : {value}")

def external_function_out(counter,value):
    print(f"out counter :  {counter}  ---  plate id : {value}")
    

passed.external_function_in = external_function_in
passed.external_function_out = external_function_out
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

path_video = "media/v3.mp4"
cap=cv2.VideoCapture(path_video)   
while True:
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, 600))
    passed.update(frame)
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
