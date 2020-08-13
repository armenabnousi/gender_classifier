import tensorflow as tf
import torchfile
import numpy as np



class VggFace():
    def __init__(self, image_size = (224,224)):
        self.image_size = image_size
        self.model = self.create_model()
        
    def create_multiple_conv2d(self, filters, x):
        for filter_count in filters:
            x = tf.keras.layers.ZeroPadding2D((1,1))(x)
            x = tf.keras.layers.Conv2D(filter_count, (3,3), use_bias = True)(x)
            x = tf.keras.layers.ReLU()(x)
        return x
        
    def create_model(self):
        input_layer = tf.keras.layers.Input(shape = (self.image_size[0], self.image_size[1], 3))
        x = self.create_multiple_conv2d([64, 64], input_layer)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = self.create_multiple_conv2d([128, 128], x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = self.create_multiple_conv2d([256, 256, 256], x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = self.create_multiple_conv2d([512, 512, 512], x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = self.create_multiple_conv2d([512, 512, 512], x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        
        x = tf.keras.layers.Conv2D(4096, (7,7))(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Conv2D(4096, (1,1))(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Conv2D(2622, (1,1))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Softmax()(x)
        
        model = tf.keras.Model(input_layer, x)
        return model
    
    def load_weights(self, t7_filepath):
        tf_layer_counter = 1
        torch_model = torchfile.load(t7_filepath)
        for i in range(len(torch_model.modules)):
            torch_layer_name = torch_model.modules[i]._typename.decode()
            if torch_layer_name in ['nn.SpatialConvolutionMM', 'nn.Linear']:
                while not self.model.layers[tf_layer_counter].name.startswith('conv2d'):
                    tf_layer_counter += 1
                tf_layer_dim = self.model.layers[tf_layer_counter].kernel_size[0]
                weight = np.swapaxes(torch_model.modules[i].weight,0,1)
                weight = [weight.reshape((-1, tf_layer_dim, tf_layer_dim, weight.shape[1]))]
                weight[0] = np.swapaxes(weight[0],0,2)
                weight.append(np.zeros(weight[0].shape[-1]))
                self.model.layers[tf_layer_counter].set_weights(weight)
                tf_layer_counter += 1
