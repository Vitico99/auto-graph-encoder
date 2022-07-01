import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from keras import layers
from .model import Model
from sklearn.model_selection import train_test_split


class NetModel(Model):
    def __init__(self, epochs=3, input_size=44, num_classes=6) -> None:
        self.epochs = epochs

        # build the NN model here using the constructor arguments
        self.model = None


    def train(self, X, Y):
        x_train,x_val,y_train,y_val = train_test_split(X, Y, test_size=0.2, random_state=13)
        train_dataset = self.vectors_to_dataset(x_train, y_train)
        val_dataset = self.vectors_to_dataset(x_val, y_val)
        self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.epochs)

    def predict(self, X):
        ax1, ax2 = X.shape
        x_predict = self.vector_to_tensor(X, (ax1, 1, ax2))
        prob_predict = self.model.predict(x_predict)
        return np.argmax(prob_predict, axis=1)


    def eval(self, X, Y):
        test_data = self.vectors_to_dataset(X,Y)
        loss, accuracy = self.model.evaluate(test_data)
        print("Loss: ", loss)
        print("Accuracy :", accuracy)

    def vector_to_tensor(self, v, shape):
        v = tf.constant(v, shape=shape)
        v = tf.data.Dataset.from_tensor_slices(v)
        return v

    def vectors_to_dataset(self, x, y):
        ax1, ax2 = x.shape

        # # transform numpy vectors in tensors of expected shape
        x = self.vector_to_tensor(x, (ax1,1,ax2))
        y = self.vector_to_tensor(y, (ax1,1))

        # zip to create the dataset
        dataset = tf.data.Dataset.zip((x,y))
        return dataset
    


class BasicNet(NetModel):
    def __init__(self, epochs=3, input_size=44, num_classes=6) -> None:
        super().__init__(epochs, input_size, num_classes)
        
        self.model = tf.keras.Sequential([
            layers.Dense(input_size, activation=None),
            layers.ReLU(),
            layers.Dense(num_classes, activation='sigmoid')
        ])

        self.model.build((1,input_size))

        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer='adam',
            metrics=keras.metrics.SparseCategoricalAccuracy()
        )
    

    

class Autoencoder(tf.keras.Model):
    def __init__(self,indim, outdim,name='autoencoder',**kwargs) -> None:
        super(Autoencoder, self).__init__(name = name, **kwargs)
        self.indim = indim
        self.outdim = outdim

        # build here the encoder 
        self.encoder = self.build_encoder()

        # build here the decoder 
        self.decoder = self.build_decoder()

    def build_encoder(self):
        raise NotImplementedError()
    
    def build_decoder(self):
        raise NotImplementedError()

    @tf.function
    def call(self, input):
        encoding = self.encoder(input)
        decoding = self.decoder(encoding)
        return decoding

    def train(self, X,epochs=10):
        ax1, ax2 = X.shape
        X = tf.constant(X,shape=(ax1,1,ax2))
        X = tf.data.Dataset.from_tensor_slices(X)
        dataset = tf.data.Dataset.zip((X,X))
        self.fit(dataset,epochs=epochs)
    
    def encode(self, X):
        ax1, ax2 = X.shape
        x_train = tf.constant(X, shape=(ax1,1,ax2))
        x_train = tf.data.Dataset.from_tensor_slices(x_train)
        y_predict = self.encoder(X)
        return y_predict
    

class SimpleAutoencoder(Autoencoder):
    def __init__(self, indim, outdim, name='simple-autoencoder', **kwargs) -> None:
        super().__init__(indim, outdim, name, **kwargs)
        self.compile(
            optimizer = tf.optimizers.Adam(0.01),
            loss = tf.losses.MeanSquaredError()
        )

    def build_encoder(self):
        return tf.keras.Sequential([
            layers.Input(shape=(self.indim,)),
            layers.Dense(self.outdim, activation='relu'),
        ])
    
    def build_decoder(self):
        return tf.keras.Sequential([
            layers.Dense(self.indim, activation='sigmoid')
        ])
    

class DeepAutoencoder(Autoencoder):
    def __init__(self, indim, outdim, name='deep-autoencoder', **kwargs) -> None:
        super().__init__(indim, outdim, name, **kwargs)

        self.compile(
            optimizer = tf.optimizers.Adam(0.01),
            loss = tf.losses.MeanSquaredError()
        )
    
    def build_encoder(self):
        return tf.keras.Sequential([
            layers.Input(shape=(self.indim,)),
            layers.Dense(self.outdim, activation='relu'),
        ])
    
    def build_decoder(self):
        return tf.keras.Sequential([
            layers.Dense(self.outdim, activation='sigmoid'),
            layers.Dense(self.indim),
        ])
    
    
    



    



