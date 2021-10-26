import json
import sys
from sklearn.metrics import accuracy_score

import utils

def load_data(data_json):
    data = data_json['results']
    cars = []
    bikes = []
    bus = []
    truck = []
    for i in range(len(data)):  
        cars.append(data[i]['objects']['car'])
        bikes.append(data[i]['objects']['bike'])
        bus.append(data[i]['objects']['bus'])
        truck.append(data[i]['objects']['truck'])

    return cars, bikes, bus, truck

if __name__ == '__main__':

    gt_json = sys.argv[1] #json with ground truth
    preds_json = sys.argv[2] #json with predictions

    gt_data = utils.load_json(gt_json)
    preds = utils.load_json(preds_json)

    gt_cars, gt_bikes, gt_bus, gt_truck = load_data(gt_data)
    pred_cars, pred_bikes, pred_bus, pred_truck = load_data(preds)

    acc_cars = accuracy_score(gt_cars, pred_cars)
    print('Cars Accuracy: %f' % acc_cars)
    acc_bikes = accuracy_score(gt_bikes, pred_bikes)
    print('Bikes Accuracy: %f' % acc_bikes)
    acc_bus = accuracy_score(gt_bus, pred_bus)
    print('Bus Accuracy: %f' % acc_bus)
    acc_truck = accuracy_score(gt_truck, pred_truck)
    print('Trucks Accuracy: %f' % acc_truck)
