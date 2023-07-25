import matplotlib.pyplot as plt
import pandas as pd
import os
import math

def plotting(x, y_1, y_2, y_3, title_1, title_2, title_3, y_label, save_file):

    plt.subplot(5,1,1)
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title_1)
    plt.scatter(x, y_1)
    plt.plot(x, y_1, color='green')
    # new_list = range(math.floor(min(y_1)), math.ceil(max(y_1)) + 1)
    # plt.yticks(new_list)
    plt.xlabel("Time")
    plt.ylabel(y_label)

    plt.subplot(5,1,3)
    plt.title(title_2)
    plt.scatter(x, y_2)
    plt.plot(x, y_2, color="orange")
    # new_list_1 = range(math.floor(min(y_2)), math.ceil(max(y_2)) + 1)
    # plt.yticks(new_list_1)
    plt.xlabel("Time")
    plt.ylabel(y_label)

    plt.subplot(5,1,5)
    plt.title(title_3)
    plt.scatter(x, y_3)
    plt.plot(x, y_3, color='red')
    # new_list_2 = range(math.floor(min(y_3)), math.ceil(max(y_3)) + 1)
    # plt.yticks(new_list_2)
    plt.xlabel("Time")
    plt.ylabel(y_label)

    plt.savefig(save_graph(save_file, y_label) + f"/{title_1}.png")
    plt.close()

    plt.title(title_1)
    plt.scatter(x, y_1)
    plt.plot(x, y_1, color='green')
    # new_list = range(math.floor(min(y_1)), math.ceil(max(y_1)) + 1)
    # plt.yticks(new_list)
    plt.xlabel("Time")
    plt.ylabel(y_label)

    plt.savefig(save_graph(save_file, y_label) + f"/{title_1}_Scaled_UP.png")
    plt.close()



def plotting_bar(label, values, title, ylabel, save_file):
    plt.bar(label, values, width=0.4)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_graph(save_file,'')+f"/{title}.png")
    plt.close()

def save_graph(file, catagory):
    dir = "/Users/thestash/PycharmProjects/Sulabh Sochalaya/Graphs/"
    try:
        os.mkdir(f"{dir}{file}/{catagory}")
        sub_dir = f"{dir}{file}/{catagory}"
        return sub_dir
    except FileExistsError:
        for i in os.listdir(dir):
            if file in i:
                for j in i:
                    if catagory in j:
                        sub_dir = catagory
                        break
        return f"{dir}{file}/{catagory}"

def analysis(file, save_file):
    df = pd.read_csv(file)
    time = []
    People = []
    Car = []
    Truck = []
    Bike = []
    Bus = []
    People_N =[]
    People_S = []
    Car_N = []
    Car_s = []
    Truck_N = []
    Truck_S = []
    Bike_N = []
    Bike_S = []
    Bus_N = []
    Bus_S = []
    t = 0
    for i in df["Time_Elapsed"]:
        t += int(i)
        time.append(t)

    for i in df["Object_Detected"]:
        row = i.split()
        ppl_count = int(row[1])
        car_count = int(row[3])
        truck_count = int(row[5])
        bike_count = int(row[7])
        bus_count = int(row[9])
        People.append(ppl_count)
        Car.append(car_count)
        Truck.append(truck_count)
        Bike.append(bike_count)
        Bus.append(bus_count)

    for i in df["Direction"]:
        row_direction = i.split()
        ppl_N = int(row_direction[1])
        ppl_S = int(row_direction[3])
        car_N = int(row_direction[5])
        car_S = int(row_direction[7])
        truck_N = int(row_direction[9])
        truck_S = int(row_direction[11])
        bike_N = int(row_direction[13])
        bike_S = int(row_direction[15])
        bus_N = int(row_direction[17])
        bus_S = int(row_direction[19])
        People_N.append(ppl_N)
        People_S.append(ppl_S)
        Car_N.append(car_N)
        Car_s.append(car_S)
        Truck_N.append(truck_N)
        Truck_S.append(truck_S)
        Bike_N.append(bike_N)
        Bike_S.append(bike_S)
        Bus_N.append(bus_N)
        Bus_S.append(bus_S)

    for i,j in zip(df["Total_Up"],df["Total_Down"]):
        pass
    tot_up = int(i)
    tot_down = int(j)
    label = ["Total Up", "Total Down"]
    values = [tot_up, tot_down]

    plotting_bar(label, values, "Objects crossed wrt Line", "Magnitude of Objects", save_file)
    plotting(time, People, People_N, People_S, "No. of People wrt Time (per sec.)", "No. of People North wrt Time", "No. of People South wrt Time" ,"People", save_file)
    plotting(time, Car, Car_N, Car_s, "No. of Cars wrt Time (per sec.)", "No. of Cars North wrt Time", "No. of Cars South wrt Time", "Cars", save_file)
    plotting(time, Truck, Truck_N, Truck_S, "No. of Trucks wrt Time (per sec.)", "No. of Trucks North wrt Time", "No. of Trucks South wrt Time", "Trucks", save_file)
    plotting(time, Bike, Bike_N, Bike_S, "No. of Bikes wrt Time (per sec.)", "No. of Bikes North wrt Time", "No. of Bikes South wrt Time", "Bikes", save_file)
    plotting(time, Bus, Bus_N, Bus_S, "No. of Busses wrt Time (per sec.)", "No. of Busses North wrt Time", "No. of Busses South wrt Time","Busses", save_file)

File = "/Users/thestash/PycharmProjects/Sulabh Sochalaya/Records/Record_video.csv"
file = File.split("/")[-1]
file = file.split(".")[0]
file = file.replace("Record", "Graph")
analysis(File, file)

