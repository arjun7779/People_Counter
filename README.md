This is a project for object counter using ML. We use the YOLOv8m.pt, and make a YOLO model using VisDrone.yaml for object detections.
The training is already done, simply download the weights from download.txt. (it is not necessarcy to use these waights, you may use your trained weights).

Make sure to give the correct path wherever necessary. places requiring the correct paths will be writtern in UpperCase.

For tracking the objects in video, SORT (Simple Online and Realtime Tracking) tracker. 

For analysing the video, just give path to corresponding csv file hence created (after running people_counter.py, csv files are stored in Records Folder) in Analysis.py. It will generate graphs and store the results in the Graphs folder.

