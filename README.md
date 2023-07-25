This is a project for object counter using ML. We use the YOLOv8m.pt, and make a YOLO model using VisDrone.yaml for object detections.
The training is already done, simply download the weights from download.txt. (it is not necessarcy to use these weights, you may use your trained weights). The pre-trained YOLOv8 model should work fine for most videos (use cases).

Make sure to give the correct path wherever necessary. places requiring the correct paths will be writtern in UpperCase in python files.

For tracking the objects in video, SORT (Simple Online and Realtime Tracking) tracker. Refer to sort.py for better understanding.

Note: The project is done on macos, using metal shaders for using GPU. For windows, to use GPU set device=0 or required device number (if GPU is available, will work with CPU as well.), in results = model_1(frame, stream=True, conf=0.3, device="mps", classes=[0,2,3,5,7]) in people_counter.py.

To Do:
1. Create empty folders Graphs, Output_Vidoes, Records or simply clone this repository. 
2. Give Correct Paths (In both people_counter.py and Analysis.py).
3. Run people_counter.py - This will do two things:
   1.This creates another video (of input video or live inference) comprising the line counter and will display the total number of objects          moving up or down the line. Then save it in Output_Vidoes in the form of PC_(video name).avi.
   2.Create a csv file in form of Record_(Video name).csv that stores the data extracted from the input video. (Time details, objects                detected, direction etc...)
4. Run Analysis.py - Copy the path of csv file made by running th video on people_counter.py and paste in Analysis.py (look at code for location). Run the code. It will generate graphs and store the results in the Graphs folder.

