import tensorflow as tf
import cv2
import numpy as np

class AdienceModel():
    def __init__(self, pretrained_vgg = None, frozen_layers = 30, add_on_top = 30, image_size = (224,224)):
        self.image_size = image_size
        if pretrained_vgg:
            pretrained = self.freeze_layers(frozen_layers, pretrained_vgg)
            self.model = self.add_layers(pretrained, add_on_top)
            
    def freeze_layers(self, frozen, pretrained_model):
        for i, layer in enumerate(pretrained_model.layers):
            if i == frozen:
                break
            layer.trainable = False
        return pretrained_model
    
    def add_layers(self, pretrained, top_layer):
        x = pretrained.get_layer(index = top_layer).output
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
        model = tf.keras.Model(pretrained.input, x)
        return model
    
    def load_model(self, filepath):
        #disable cpu warning
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        #load the model
        self.model = tf.keras.models.load_model(filepath)
     
    def classify_single_image(self, img):
        img = cv2.resize(cv2.imread(img), self.image_size)
        img = np.expand_dims(img, 0) / 255
        output = self.model.predict(img)[0][0]
        output = (output, 'female') if output < 0.5 else (output, 'male')
        return output
