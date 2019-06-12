from model.multi_trainer import Multi_Trainer as L
import sys
import requests
import json


class trainer_func:
    def __init__(self):
        self.train = L(
            save_dir="./static/Own_classes/save",
            train_dir="./static/Own_classes/train",
            json_dir="./model/"
        )
        self.startargs = sys.argv[1:]
    
    def callback(self, data, header):
        root='http://localhost:5000'
        path='/trainprogress'
        requests.post(root + path, json=data, headers=header)

    def live(self):
        data = {
            'epoch': -1,
            'val_loss': 0,
            'val_acc': 0,
            'loss': 0,
            'acc': 0
        }
        header = {'wooden': 'STARTED!!!'}
        self.callback(data, header)
        if "classifajar" in self.startargs:
            self.train.train_Classifajar()
        if "uppercase" in self.startargs:
            self.train.train_uppercase()
        if "lowercase" in self.startargs:
            self.train.train_lowercase()
        if "numbers" in self.startargs:
            self.train.train_numbers()
        header = {'wooden': "FINISHED!!!"}
        self.callback(data, header)

t = trainer_func()
t.live()
