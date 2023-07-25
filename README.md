This is a basic project for people counter using AI/ML. We use the YOLOv8 model and futher train it using the VisDrone data set. 
The training is already done, simply download the weights from download.txt. (it is not necessarcy to use these waights, you may use your trained weights)
Make sure to give the correct path to you video dataset in video = cv.VideoCapture("path"), the correct path to the downloaded weights to YOLO in people_counter.py. Similary, give correct paths as per requirements all python files.

For tracking the objects in video, SORT (Simple Online and Realtime Tracking) tracker. 
For analysing the video, just give path to corresponding csv file hence created (after running people_counter.py, csv files are stored in Records Folder) in Analysis.py. It will generate graphs and store the results in the Graphs folder.

