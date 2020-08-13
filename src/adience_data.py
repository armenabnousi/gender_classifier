import tensorflow as tf
import cv2
import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing 

class AdienceData():
    def __init__(self, indir, datatype = "train", image_size = (224, 224), 
                 shuffle = None, batch_size = 32, augment = None, memory = "low"):
        self.shuffle = shuffle if shuffle is not None else True if datatype == "train" else False
        self.augment = augment if augment is not None else True if datatype == "train" else False
        self.batch_size = batch_size
        self.image_size = image_size
        if augment:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=10, width_shift_range=0.1,
                    height_shift_range=0.1, brightness_range=[0.95, 1.05], 
                    shear_range=0.01, zoom_range=0.05,
                    vertical_flip=True, rescale = 1./255)
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
        if memory == "high":
            self.data, self.x, self.y, self.datagen = self.load_data_for_high_memory(indir, datatype,
                                                                                image_size, batch_size, 
                                                                                shuffle, datagen)
        else:
            self.data, self.datagen = self.load_data_for_low_memory(indir, datatype, image_size, 
                                                                    batch_size, shuffle, datagen)
            
            
            
    def load_data_for_low_memory(self, indir, datatype, image_size, batch_size, shuffle, datagen):
        data = self.load_data(indir, datatype, image_size, load_type = "filepath")
        datagen = datagen.flow_from_dataframe(data, x_col = "filename", 
                                              y_col = "gender", target_size = image_size,
                                              class_mode = "binary", batch_size=batch_size, 
                                              shuffle=shuffle)
        return data, datagen
            
    def load_data_for_high_memory(self, indir, datatype, image_size, batch_size, shuffle, datagen):
        data = self.load_data(indir, datatype, image_size, load_type = "binary")
        x = np.array(data['img'].to_list())
        y = data['gender_binary'].tolist()
        dataflow = datagen.flow(x, y, batch_size=batch_size, shuffle=shuffle)
        return data, x, y, dataflow
            
        
    def load_data(self, indir, datatype = "train", image_size = (224, 224), load_type = "binary"):
        subdir = "aligned" if datatype == "train" else "valid"
        data_dir = glob.glob(f'{os.path.join(indir, subdir)}/*_[FM]')
        d = self.load_images(data_dir, image_size = image_size, load_type = load_type)
        d['gender_binary'] = preprocessing.LabelBinarizer().fit_transform(d['gender'])
        d = d.sample(frac=1).reset_index(drop = True)
        return d
    
    def load_images(self, indir, image_size = (224, 224), load_type = "binary"):
        imgs = []
        ages = []
        names = []
        genders = []
        for directory in indir:
            filenames = os.listdir(directory)
            age = int(directory[(directory.rfind('/')+1):(directory.rfind('_'))])
            gender = 'female' if directory[(directory.rfind('_') + 1):] == 'F' else 'male' if \
                                        directory[(directory.rfind('_') + 1):] == 'M' else 'N'
            ages.extend([age] * len(filenames))
            genders.extend([gender] * len(filenames))
            if load_type == "binary":
                names.extend(filenames)
                for filename in filenames:
                    img = cv2.imread(os.path.join(directory, filename))
                    img = cv2.resize(img, image_size)
                    imgs.append(img)
            else:
                filenames = [os.path.join(directory, filename) for filename in filenames]
                names.extend(filenames)
        if load_type == "binary":
            d = pd.DataFrame({'filename':names,'gender':genders, 'age':ages, 'img':imgs})
        else:
            d = pd.DataFrame({'filename':names,'gender':genders, 'age':ages})
        return d
