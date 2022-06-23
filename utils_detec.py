import json


def load_json(data):
    with open(data) as f:
        data_json = json.load(f)
    return data_json


def save_json(path_json, data):
    with open(path_json, "w") as outfile:
        json.dump(data, outfile)
    return


def build_monitor_cpu_results(cpu_u, cpu_f, ram_u):

    cpu_usage = sum(cpu_u) / len(cpu_u)
    cpu_freq = sum(cpu_f) / len(cpu_f)
    ram_usage = sum(ram_u) / len(ram_u)

    return cpu_usage, cpu_freq, ram_usage


def build_monitor_gpu_results(gpuUtil, memUtil):

    gpu_util = sum(gpuUtil) / len(gpuUtil)
    mem_util = sum(memUtil) / len(memUtil)

    return gpu_util, mem_util


def build_results(
    total_frames,
    total_time_frames,
    cars_frame,
    bus_frame,
    truck_frame,
    bikes_frame,
    num_car,
    num_bike,
    num_bus,
    num_truck,
):

    data = {}
    data["results"] = []
    for i in range(len(total_frames)):
        # data["results"][i] = {}
        data["results"].append({"id": total_frames[i]})
        data["results"][i]["time_frame"] = total_time_frames[i]
        data["results"][i]["objects"] = {
            "car": cars_frame[i],
            "truck": truck_frame[i],
            "bus": bus_frame[i],
            "bike": bikes_frame[i],
            "parked": 0,
        }

    total_vehicles = num_car + num_bike + num_bus + num_truck

    print(total_vehicles)
    if total_vehicles > 0:
        per_car = round(num_car / total_vehicles, 2)
        per_bike = round(num_bike / total_vehicles, 2)
        per_bus = round(num_bus / total_vehicles, 2)
        per_truck = round(num_truck / total_vehicles, 2)
    else:
        per_car = 0
        per_bike = 0
        per_bus = 0
        per_truck = 0

    res = {}
    res["total_vehicles"] = total_vehicles
    res["type"] = []
    res["type"].append({"cars": num_car, "percentage": per_car})
    res["type"].append({"bikes": num_bike, "percentage": per_bike})
    res["type"].append({"buses": num_bus, "percentage": per_bus})
    res["type"].append({"trucks": num_truck, "percentage": per_truck})
    # print(res)

    # data["resume"] = res #to join both dict on the same json

    return data, res


def count_vehicles(df, num_car, num_bike, num_bus, num_truck):

    cuenta = df["labels"]
    for v in cuenta:
        if v == "car":
            num_car += 1
        if v == "motorcycle":
            num_bike += 1
        if v == "bus":
            num_bus += 1
        if v == "truck":
            num_truck += 1

    return num_car, num_bike, num_bus, num_truck


def count_df_vehicle_types(df, type, cars, bikes, buses, trucks):

    for i in range(0, len(df[type])):
        if df[type][i]["type"] == "car":
            cars += 1
        if df[type][i]["type"] == "motorcycle":
            bikes += 1
        if df[type][i]["type"] == "bus":
            buses += 1
        if df[type][i]["type"] == "truck":
            trucks += 1

    return cars, bikes, buses, trucks


def count_vehicles_moving(df, output_path, treshold=100):

    # To control the number of vehicles NOT parked
    num_car = 0
    num_bike = 0
    num_bus = 0
    num_truck = 0

    total_frames = len(df["id_frame"].unique())
    print("TOTAL FRAMES: ", total_frames)

    tracked_ids = df["id_track"].unique()
    tracked_ids = tracked_ids[1:]

    moving_vehicles = []
    parked_vehicles = []
    for t in tracked_ids:
        # print(len(df[df["id_track"] == t]))
        if len(df[df["id_track"] == t]) < treshold:
            moving_vehicles.append(t)
        else:
            # print("Parked vehicles ID: ", t)
            parked_vehicles.append(t)

    new_df = df[df["id_track"].isin(moving_vehicles)]

    cuenta_x_frame = new_df["labels"]

    for i, v in enumerate(cuenta_x_frame):
        if v == "car":
            num_car += 1
        if v == "motorcycle":
            num_bike += 1
        if v == "bus":
            num_bus += 1
        if v == "truck":
            num_truck += 1

    n = 1
    counts = {}
    counts["parked_vehicles"] = []
    counts["moving_vehicles"] = []

    for m in parked_vehicles:
        aux = df[df["id_track"] == m]
        cuenta = aux["labels"].value_counts()[:n].index.tolist()
        counts["parked_vehicles"].append({"id": int(m), "type": cuenta[0]})

    for m in moving_vehicles:
        aux = df[df["id_track"] == m]
        cuenta = aux["labels"].value_counts()[:n].index.tolist()
        counts["moving_vehicles"].append({"id": int(m), "type": cuenta[0]})

    cars_parked = bikes_parked = buses_parked = trucks_parked = 0
    cars_parked, bikes_parked, buses_parked, trucks_parked = count_df_vehicle_types(
        counts,
        "parked_vehicles",
        cars_parked,
        bikes_parked,
        buses_parked,
        trucks_parked,
    )

    cars_moving = bikes_moving = buses_moving = trucks_moving = 0
    cars_moving, bikes_moving, buses_moving, trucks_moving = count_df_vehicle_types(
        counts,
        "moving_vehicles",
        cars_moving,
        bikes_moving,
        buses_moving,
        trucks_moving,
    )

    counts["total_moving_vehicles"] = {
        "num": len(moving_vehicles),
        "cars": cars_moving,
        "motorcycles": bikes_moving,
        "buses": buses_moving,
        "trucks": trucks_moving,
    }

    counts["total_parked_vehicles"] = {
        "num": len(parked_vehicles),
        "cars": cars_parked,
        "motorcycles": bikes_parked,
        "buses": buses_parked,
        "trucks": trucks_parked,
    }

    # print(counts)

    out = output_path.split(".")[0]
    with open(out + "_resumen.txt", "w") as f:
        f.write("TOTAL MOVING VEHICLES: " + str(len(moving_vehicles)) + "\n")
        f.write("------------------------------------------\n")
        f.write("Total cars: " + str(cars_moving) + "\n")
        f.write("Total motorcycles: " + str(bikes_moving) + "\n")
        f.write("Total buses: " + str(buses_moving) + "\n")
        f.write("Total trucks: " + str(trucks_moving) + "\n")
        f.write("\n")
        f.write("TOTAL PARKED VEHICLES: " + str(len(parked_vehicles)) + "\n")
        f.write("------------------------------------------\n")
        f.write("Total cars: " + str(cars_parked) + "\n")
        f.write("Total motorcycles: " + str(bikes_parked) + "\n")
        f.write("Total buses: " + str(buses_parked) + "\n")
        f.write("Total trucks: " + str(trucks_parked) + "\n")

    print("TOTAL MOVING VEHICLES: ", len(moving_vehicles))
    print("------------------------------------------")
    print("Total cars: ", cars_moving)
    print("Total motorcycles: ", bikes_moving)
    print("Total buses: ", buses_moving)
    print("Total trucks: ", trucks_moving)

    print("TOTAL PARKED VEHICLES: ", len(parked_vehicles))
    print("------------------------------------------")
    print("Total cars: ", cars_parked)
    print("Total motorcycles: ", bikes_parked)
    print("Total buses: ", buses_parked)
    print("Total trucks: ", trucks_parked)

    return num_car, num_bike, num_bus, num_truck, counts
