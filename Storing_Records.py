import shutil
import ast
import datetime
import csv

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

def keep_record(Time_Elapsed, Frame_Elasped, Direction=None, Objects_Detected=None, Total_Up=None, Total_Down=None):
    TIME = datetime.datetime.now()
    Date = TIME.date()
    Day = TIME.strftime("%A")
    Time = TIME.time()
    with open("record.csv", "a") as file:
        csv_file = csv.writer(file)
        csv_file.writerow([Date, Day, Time, Time_Elapsed, Frame_Elasped, Objects_Detected, Direction, Total_Up, Total_Down])

def Store(dir, file):
    shutil.copyfile("record.csv", dir + f"/Record_{file}.csv")
    return dir + f"/Record_{file}.csv"









