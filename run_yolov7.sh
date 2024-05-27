#!/bin/bash

# Navigate to the directory for the first command
cd /home/hynduf/yolov7

# Run the first command
/tmp/yolov7/python detect.py --weights yolov7-tiny.pt --conf 0.4 --source 0 --class 0
