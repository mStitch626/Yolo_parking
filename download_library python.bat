@echo off

python.exe -m pip install --upgrade pip

for %%x in (
numpy
scipy
pandas
matplotlib
opencv-python
PyQt5
keras
torch
tensorflow
mediapipe
ultralytics
) do (
 echo download %%x
 pip install %%x
 echo ---------------------------------------------
)
echo 'end' 
pause