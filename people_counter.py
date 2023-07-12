from ultralytics import YOLO
import cv2 as cv
import cvzone
from sort import*
import math
import ast
from RES_ADJ import*


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


video = cv.VideoCapture("path to video")

#Use the model which suits the use case
#give path to model of weight bestm.pt
model = YOLO("/Users/thestash/PycharmProjects/Sulabh Sochalaya/visdroneWeights/bestm.pt")
model_1 = YOLO("yolov8m.pt")

width = int(video.get(3))
height = int(video.get(4))


x = res_adjustment(width, height)[0]
y = res_adjustment(width, height)[1]
subtract = res_adjustment(width, height)[2]
Scale = res_adjustment(width, height)[3]
Thickness = res_adjustment(width, height)[4]



tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)
line_cords = ((0,int(height/2)),(int(width),int(height/2)))


people_counter = []
people_counter_up = []
people_counter_down = []
counter = 0

previous_y = None

fourcc = cv.VideoWriter_fourcc(*'XVID')
video_output = cv.VideoWriter("/Users/thestash/PycharmProjects/Sulabh Sochalaya/Output_Vidoes/testing.avi", fourcc, 30, (width, height))


def read_PY(file, key):
    with open(file, 'r') as f:
        lines = f.readlines()
        memory_ele = None
        for x in lines:
            if x == "\n":
                continue
            y = ast.literal_eval(x)
            if y.keys() == key.keys():
                memory_ele = y
        f.close()
    return memory_ele

def change_PY(file, track_py):
    with open(file, 'a') as f:
        f.write(track_py)
        f.close()


while video.isOpened():
    var, frame = video.read()

    #model will be changed here model or model_1
    results = model(frame, stream=True, conf=0.3, device="mps") #remove classes later
    cv.line(frame, line_cords[0], line_cords[1], (0,255,0), thickness=4)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            catagory = int(box.cls[0])


            cv.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), thickness=1)
            #for displaying catagory, and conficence. Commented for the sake of neatness on the video
            #cvzone.putTextRect(frame,class_names[catagory]+" "+str(confidence), (x1,y1-10), scale=0.8, thickness=0)


            currentArray = np.array([x1, y1, x2, y2, confidence])
            detections = np.vstack((detections, currentArray))

    result_tracked = tracker.update(detections)


    for t_results in result_tracked:
        x1, y1, x2, y2, tracking_id = t_results
        x1, y1, x2, y2, tracking_id = int(x1), int(y1), int(x2), int(y2), int(tracking_id)
      
        #For displaying tracking id on the video
        #cvzone.putTextRect(frame, str(tracking_id), (x1, y1 - 30), scale=0.8, thickness=1)

        center_x, center_y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        track_var_py = {tracking_id: previous_y}
        track_var_x = {tracking_id: center_x}
        track_var_y = {tracking_id: center_y}


        if tracking_id not in people_counter:
            people_counter.append(tracking_id)
            previous_y = track_var_y[tracking_id]
            track_var_py[tracking_id] = previous_y
            change_PY("record.txt", str(f"\n{track_var_py}"))


        if line_cords[0][0] <= track_var_x[tracking_id] <= line_cords[1][0] and track_var_y[tracking_id] in range(line_cords[0][1] - 20, line_cords[0][1] + 20):

            if read_PY("record.txt", track_var_y)[tracking_id] < track_var_y[tracking_id]:
                if tracking_id not in people_counter_up:
                        cv.line(frame, line_cords[0], line_cords[1], (0, 0, 255), thickness=5)
                        people_counter_up.append(tracking_id)
                        previous_y = int(height/2)

            elif read_PY("record.txt", track_var_y)[tracking_id] > track_var_y[tracking_id]:
                if tracking_id not in people_counter_down:
                        cv.line(frame, line_cords[0], line_cords[1], (255, 0, 0), thickness=5)
                        people_counter_down.append(tracking_id)
                        previous_y = int(height/2)


    cvzone.putTextRect(frame, "DOWN: "+ str(len(people_counter_up)), (x, y), scale=Scale, colorR=(255,0,0), thickness=Thickness)
    cvzone.putTextRect(frame, "UP: "+ str(len(people_counter_down)), (width-subtract, y), scale=Scale, colorR=(255,0,0), thickness=Thickness)

    cv.imshow("counter", frame)
    video_output.write(frame)
    if cv.waitKey(1) == ord("q"):
        break



open("record.txt", "w").close()
video_output.release()
video.release()
cv.destroyAllWindows()



