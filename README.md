This is a basic project for people counter using AI/ML. We use the YOLOv8 model and futher train it using the VisDrone data set. 
The training is already done, simply download the weights from download.txt. (it is not necessarcy to use these waights, you may use your trained weights)
Make sure to give the correct path to you video dataset in video = cv.VideoCapture("path"), the correct path to the downloaded weights to YOLO. 


For tracking the objects in video, SORT (Simple Online and Realtime Tracking) tracker. 


Note: After running a video sucessfully, you may have to delete all data from record.txt manually before running a new video. (This will be patched and updated soon).
