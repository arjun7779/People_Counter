from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from RES_ADJ import*
from Storing_Records import*
from sort import*
import time

class_names = [
  "pedestrian",
  'people',
  'bicycle',
  'car',
  'van',
  'truck',
  'tricycle',
  'awning-tricycle',
  'bus',
  'motor']

class_names_1 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#path to the video/live camera
File = "FULL PATH TO VIDEO"
file = File.split("/")[-1]
file = file.split(".")[0] #for creating folder names of csv files.

video = cv.VideoCapture(File)
model = YOLO("PATH TO VISDRONE WEIGHTS/BESTM.PT")
model_1 = YOLO("yolov8m.pt")
width = int(video.get(3))
height = int(video.get(4))

#see RES_ADJ.py, script that adjusts the scale, thickness of line, bounding boxes etc.. on the video/live inference
x = res_adjustment(width, height)[0]
y = res_adjustment(width, height)[1]
subtract = res_adjustment(width, height)[2]
Scale = res_adjustment(width, height)[3]
Thickness = res_adjustment(width, height)[4]

#initialize the SORT tracker
tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.5)
#defining the location of line on the video
line_cords = ((0,int(height/2)),(int(width),int(height/2)))

# initializing variables/lists for stroing values
Total_counter = [] #same as frame_counter but for line only
Total_counter_up = []
Total_counter_down = []

people_counter_North = []
people_counter_South = []
people_count = []

car_counter_North = []
car_counter_South = []
car_count = []

motorcycle_counter_North = []
motorcycle_counter_South = []
motorcycle_count = []

bus_counter_North = []
bus_counter_South = []
bus_count = []

Truck_counter_North = []
Truck_counter_South = []
Truck_count = []

frame_counter = [] #throught the frame
frame_elapsed = 0

time_counter = []

#stores the y coordinate in the initial frame when the tracked object first appears
previous_y = None

#setting timer, for calculating results and storing periodically after specific time
start_time = time.time()

#creating a object to store video with line counter in memory
fourcc = cv.VideoWriter_fourcc(*'XVID')
video_output = cv.VideoWriter(f"PATH TO OUTPUT_VIDEOS FOLDER/PC_{file}.avi", fourcc, 30, (width, height))

dir = "PATH TO RECORDS FOLDER"

#creating a csv files with mentioned columns for data we want to extract and store from video.
with open("record.csv", "w") as files:
    csv_file = csv.writer(files)
    csv_file.writerow(['Date', "Day", "Time", "Time_Elapsed", "Frame_Elapsed", 'Object_Detected', 'Direction','Total_Up','Total_Down'])

while video.isOpened():
    try:
        #measuring time elapsed
        current_time = time.time()
        elapsed_time = current_time - start_time

        var, frame = video.read()
        frame_elapsed += 1
        #taking each frame and passing it via the model(neural network) for obtaining detections
        results = model_1(frame, stream=True, conf=0.3, device="mps", classes=[0,2,3,5,7])
        #displaying the line on the video
        cv.line(frame, line_cords[0], line_cords[1], (0,255,0), thickness=4)

        #numpy array that stores tracking id, coordinates (xyxy) on the frame of the objects detected
        detections = np.empty((0, 5))
        #extractiog coordinates of detections and then using them in open cv to display rectangles
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                catagory = int(box.cls[0])
                #CLASS = class_names[catagory]
                CLASS = class_names_1[catagory]

                cv.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), thickness=2)

                currentArray = np.array([x1, y1, x2, y2, confidence])
                #updating the tracking varable with ne detection and their tracking id, coordinates
                detections = np.vstack((detections, currentArray))
        #storing all the tracked objects and their id, coordinates in one variable
        result_tracked = tracker.update(detections)

        #using
        for t_results in result_tracked:
            x1, y1, x2, y2, tracking_id = t_results
            x1, y1, x2, y2, tracking_id = int(x1), int(y1), int(x2), int(y2), int(tracking_id)
            #cvzone.putTextRect(frame, str(tracking_id)+CLASS, (x1, y1 - 30), scale=0.8, thickness=1)

            #taking the centroid of each bounding box
            center_x, center_y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            
            #storing in a dictionary in the form of {tracking_id: required_coordinate}
            track_var_py = {tracking_id: previous_y}
            track_var_x = {tracking_id: center_x}
            track_var_y = {tracking_id: center_y}

            #resolving the class conflict we get as we switch between two different models
            if CLASS == 'pedestrian':
                CLASS = 'person'
            elif CLASS == 'bicycle' or CLASS == 'motor':
                CLASS = 'motorcycle'
            track_var_cls = {tracking_id: CLASS}
            
            #for updating the variable previous_y and storing in record.txt, look at Storing_Records.py
            if tracking_id not in Total_counter:
                Total_counter.append(tracking_id)
                previous_y = track_var_y[tracking_id]
                track_var_py[tracking_id] = previous_y
                change_PY("record.txt", str(f"\n{track_var_py}"))

            #checking for objects found in particular frame, keeping track of them individually
            if tracking_id not in time_counter:
                time_counter.append(tracking_id)
                if track_var_cls[tracking_id] == 'person':
                    people_count.append(tracking_id)
                elif track_var_cls[tracking_id] == 'car':
                    car_count.append(tracking_id)
                elif track_var_cls[tracking_id] == 'truck':
                    Truck_count.append(tracking_id)
                elif track_var_cls[tracking_id] == 'bus':
                    bus_count.append(tracking_id)
                elif track_var_cls[tracking_id] == 'motorcycle':
                    motorcycle_count.append(tracking_id)
                    
            # checking for objects found in particular frame, which are moving South, keeping track of them individually
            if read_PY("record.txt", track_var_y)[tracking_id] < track_var_y[tracking_id]:
                if tracking_id not in frame_counter:
                    frame_counter.append(tracking_id)
                    if track_var_cls[tracking_id] == 'person':
                        people_counter_South.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'car':
                        car_counter_South.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'truck':
                        Truck_counter_South.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'bus':
                        bus_counter_South.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'motorcycle':
                        motorcycle_counter_South.append(tracking_id)
                        
            # checking for objects found in particular frame, which are moving North, keeping track of them individually
            elif read_PY("record.txt", track_var_y)[tracking_id] > track_var_y[tracking_id]:
                if tracking_id not in frame_counter:
                    frame_counter.append(tracking_id)
                    if track_var_cls[tracking_id] == 'person':
                        people_counter_North.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'car':
                        car_counter_North.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'truck':
                        Truck_counter_North.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'bus':
                        bus_counter_North.append(tracking_id)
                    elif track_var_cls[tracking_id] == 'motorcycle':
                        motorcycle_counter_North.append(tracking_id)

            #check for all objects crossing the line
            if line_cords[0][0] <= track_var_x[tracking_id] <= line_cords[1][0] and track_var_y[tracking_id] in range(line_cords[0][1] - 20, line_cords[0][1] + 20):
                # when object moves South
                if read_PY("record.txt", track_var_y)[tracking_id] < track_var_y[tracking_id]:
                    if tracking_id not in Total_counter_up:
                        if CLASS == 'person' or CLASS == 'car' or CLASS == 'truck' or CLASS == 'bus' or CLASS == 'motorcycle':
                                cv.line(frame, line_cords[0], line_cords[1], (0, 0, 255), thickness=5)
                                Total_counter_up.append(tracking_id)
                                previous_y = int(height/2)
                #when object moves north
                elif read_PY("record.txt", track_var_y)[tracking_id] > track_var_y[tracking_id]:
                    if tracking_id not in Total_counter_down:
                        if CLASS == 'person' or CLASS == 'car' or CLASS == 'truck' or CLASS == 'bus' or CLASS == 'motorcycle':
                            cv.line(frame, line_cords[0], line_cords[1], (255, 0, 0), thickness=5)
                            Total_counter_down.append(tracking_id)
                            previous_y = int(height / 2)
                            
        #storing all the computations in csv file as one second has elapsed, refer to Storing_records.py
        if elapsed_time >= 1.0:
            keep_record(Time_Elapsed=elapsed_time, Frame_Elasped=frame_elapsed, Objects_Detected= f'''People: {len(people_count)} Cars: {len(car_count)} Truck: {len(Truck_count)} Bike: {len(motorcycle_count)} Bus: {len(bus_count)} ''',
                        Direction=f'''People_North: {len(people_counter_North)} People_South: {len(people_counter_South)} 
Car_North: {len(car_counter_North)} Car_South: {len(car_counter_South)} 
Truck_North: {len(Truck_counter_North)} Truck_South: {len(Truck_counter_South)}
Bike_North: {len(motorcycle_counter_North)} Bike_South: {len(motorcycle_counter_South)}
Bus_North: {len(bus_counter_North)} Bus_South: {len(bus_counter_South)}''',
                        Total_Up=len(Total_counter_down), Total_Down=len(Total_counter_up))
            #resetting the variables/list as after one second, we recompute results
            start_time = time.time()
            frame_elapsed = 0
            time_counter = []
            frame_counter = []
            people_counter_North = []
            people_counter_South = []
            car_counter_North = []
            car_counter_South = []
            motorcycle_counter_North = []
            motorcycle_counter_South = []
            bus_counter_North = []
            bus_counter_South = []
            Truck_counter_North = []
            Truck_counter_South = []
            people_count = []
            car_count = []
            motorcycle_count = []
            bus_count = []
            Truck_count = []
        
        #displaying total going up or down the line on live inference/video
        cvzone.putTextRect(frame, "DOWN: "+ str(len(Total_counter_up)), (x, y), scale=Scale, colorR=(255,0,0), thickness=Thickness)
        cvzone.putTextRect(frame, "UP: "+ str(len(Total_counter_down)), (width-subtract, y), scale=Scale, colorR=(255,0,0), thickness=Thickness)

        #displaying the video/live inference itself
        cv.imshow("counter", frame)
        video_output.write(frame)
        if cv.waitKey(1) == ord("q"):
            break
    except FileNotFoundError:
        break

#Storing the results from record.csv into separate csv file, refer to Storing_Records.py
Store(dir, file)

#deleteing content of record.txt, record.csv for next run
open("record.txt", "w").close()
open("record.csv", "w").close()

video_output.release()
video.release()
cv.destroyAllWindows()



