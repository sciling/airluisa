import json

def load_json(data):
    with open(data) as f: 
        data_json = json.load(f) 
    return data_json

def save_json(path_json,data):
    with open(path_json, 'w') as outfile:
        json.dump(data, outfile)
    return

def build_results(total_frames, total_time_frames, cars_frame, bus_frame, truck_frame, bikes_frame, 
                    num_car, num_bike, num_bus, num_truck):

    data = {}
    data['results'] = []
    for i in range(len(total_frames)):
        #data['results'][i] = {}
        data['results'].append({'id':total_frames[i]})
        data['results'][i]['time_frame'] = total_time_frames[i]
        data['results'][i]['objects'] = {"car": cars_frame[i] ,"truck": truck_frame[i], "bus": bus_frame[i], "bike": bikes_frame[i], "parked": 0}


    total_vehicles = num_car + num_bike + num_bus + num_truck

    print(total_vehicles)
    per_car = round(num_car/total_vehicles, 2)
    per_bike = round(num_bike/total_vehicles, 2)
    per_bus = round(num_bus/total_vehicles, 2)
    per_truck = round(num_truck/total_vehicles, 2)

    res = {}
    res['total_vehicules'] = total_vehicles
    res['type'] = []
    res['type'].append({'cars':num_car, 'percentage': per_car})
    res['type'].append({'bikes':num_bike, 'percentage': per_bike})
    res['type'].append({'buses':num_bus, 'percentage': per_bus})
    res['type'].append({'trucks':num_truck, 'percentage': per_truck})
    print(res) 

    #data['resume'] = res #to join both dict on the same json

    return data, res

def count_vehicles(df, num_car, num_bike, num_bus, num_truck):

    cuenta = df['labels']
    for v in cuenta:
        if v == "car":
            num_car += 1
        if v == "motorbike":
            num_bike += 1
        if v == "bus":
            num_bus += 1
        if v == "truck":
            num_truck += 1
    
    return num_car, num_bike, num_bus, num_truck

def count_vehicles_moving(df, treshold = 300):

    #To control the number of vehicles NOT parked
    num_car = 0
    num_bike = 0
    num_bus = 0
    num_truck = 0

    total_frames = len(df['id_frame'].unique())
    print("TOTAL FRAMES: ", total_frames)

    tracked_ids = df['id_track'].unique()
    tracked_ids = tracked_ids[1:]

    moving_vehicles = []
    parked_vehicles = []
    for t in tracked_ids:
        #print(len(df[df['id_track'] == t]))
        if len(df[df['id_track'] == t]) < treshold:
            moving_vehicles.append(t)
        else:
            print("Parked vehicles ID: ", t)
            parked_vehicles.append(t)

    print("Total moving vehicles: ", len(moving_vehicles))
    print("Total parked vehicles: ", len(parked_vehicles))

    new_df = df[df['id_track'].isin(moving_vehicles)]

    cuenta = new_df['labels']

    for i,v in enumerate(cuenta):
        if v == "car":
            num_car += 1
        if v == "motorbike":
            num_bike += 1
        if v == "bus":
            num_bus += 1
        if v == "truck":
            num_truck += 1

    return num_car, num_bike, num_bus, num_truck