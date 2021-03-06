import numpy
import requests
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
import json
if __name__ == "__main__":
    from train_former import Train_Former as Former
else:
    from model.train_former import Train_Former as Former
    
import matplotlib.pyplot as plt


class Multi_Trainer(Former):

    seed = 7
    numpy.random.seed(seed)
    source_dir = None
    img_rows, img_cols = 42, 42
    input_shape = (img_rows, img_cols, 1)
    num_classes = None
    
    def __init__(
        self,
        save_dir="../static/Own_classes/save",
        train_dir="../static/Own_classes/train",
        json_dir="./",
        uppercase="/uppercase",
        lowercase="/lowercase",
        numbers="/numbers",
        classifajar="/Classifajar"
    ):
        super(Multi_Trainer, self).__init__(
            save_dir,
            train_dir,
            json_dir,
            uppercase,
            lowercase,
            numbers,
            classifajar
        )
    
    def check_if_data(self):
        res = True
        all_data = self.accountant(True)
        train_data = all_data.get("Train_dir")
        for key, value in train_data.items():
            number = value.get('Min_fc', 0)
            if number <= 0:
                res = False
                break
        return res
        
    def train_Classifajar(self):
        Multi_Trainer.source_dir = self.train_class
        Multi_Trainer.num_classes, dir_list = Former.count_dir(
            Multi_Trainer.source_dir)
        self.modeler("Classifajar", 400)
        return print("Source dir: {}, num_classes: {}".format(
            Multi_Trainer.source_dir, Multi_Trainer.num_classes))

    def train_uppercase(self):
        Multi_Trainer.source_dir = self.train_upper
        Multi_Trainer.num_classes, dir_list = Former.count_dir(
            Multi_Trainer.source_dir)
        self.modeler("uppercase")
        return print("Source dir: {}, num_classes: {}".format(
            Multi_Trainer.source_dir, Multi_Trainer.num_classes))

    def train_lowercase(self):
        Multi_Trainer.source_dir = self.train_lower
        Multi_Trainer.num_classes, dir_list = Former.count_dir(
            Multi_Trainer.source_dir)
        self.modeler("lowercase")
        return print("Source dir: {}, num_classes: {}".format(
            Multi_Trainer.source_dir, Multi_Trainer.num_classes))

    def train_numbers(self):
        Multi_Trainer.source_dir = self.train_numbr
        Multi_Trainer.num_classes, dir_list = Former.count_dir(
            Multi_Trainer.source_dir)
        self.modeler("numbers")
        return print("Source dir: {}, num_classes: {}".format(
            Multi_Trainer.source_dir, Multi_Trainer.num_classes))

    def visualise(self, train):
        print(train.__dict__)
        plt.figure(figsize=(6,6))
        plt.imshow(train[1])
        plt.title(train[1].argmax())
        
    def modeler(self, name, add_dense=0):
        visualize = callbacks.RemoteMonitor(
            root='http://localhost:5000',
            path='/trainprogress',
            field='data',
            headers={'wooden': name},
            send_as_json=True
        )
        opti_stop = callbacks.EarlyStopping(
            monitor='val_acc',
            min_delta=0.01,
            patience=10,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
        train, test = self.generators()
        model = self.large_model(add_dense)
        model.summary()
        train_map = train.class_indices
        file = self.json_dir + "/models_multi/labels_{}.json".format(name)
        with open(file, 'w') as f:
            json.dump(train_map, f)
        print(train_map)
        model.fit_generator(
            train,
            steps_per_epoch=100,
            epochs=20,
            validation_data=test,
            validation_steps=100,
            callbacks=[visualize, opti_stop]
        )
        model.save(self.json_dir + '/models_multi/model_{}.h5'.format(name))
#        scores = model.evaluate_generator(model, test, steps=100)
        del model
#        print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))
        return print("Done")

    def generators(self):
        data = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        train = data.flow_from_directory(
            Multi_Trainer.source_dir,
            target_size=(42, 42),
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=20,
            subset="training")
        test = data.flow_from_directory(
            Multi_Trainer.source_dir,
            target_size=(42, 42),
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=4,
            subset="validation")
        return train, test

    def large_model(self, add_dense=0):
        model = Sequential()
        ################### KAGGLE ##################################
        model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=Multi_Trainer.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32,kernel_size=3,activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(64,kernel_size=3,activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64,kernel_size=3,activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128 + add_dense, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(Multi_Trainer.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        #################################################################
        
        return model

if __name__ == "__main__":
    avinas = Multi_Trainer()
    avinas.train_Classifajar()
    avinas.train_uppercase()
    avinas.train_lowercase()
    avinas.train_numbers()


