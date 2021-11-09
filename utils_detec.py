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
