from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense,Lambda
from keras.layers.core import Dropout
from keras.callbacks import Callback
from keras.layers import Reshape, BatchNormalization
from keras import backend as K
import keras

class Model():
	def __init__(self, width, height, depth, classes):
		self.width =width
		self.height= height
		self.depth = depth
		self.classes = classes
		
	def build(self):
	    # initialize the model
	    model = Sequential()
	    inputShape = (self.height, self.width, self.depth)
	   
	    # first set of CONV => RELU => POOL layers
	    model.add(Conv2D(32, (3, 3), padding="same",
		input_shape=inputShape))
	    
	    model.add(Activation("relu"))
	    model.add(BatchNormalization())
	    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	

	    # second set of CONV => RELU => POOL layers
	 #   model.add(Conv2D(32, (3, 3), padding="same"))
	  #  model.add(Activation("relu"))
	  #  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	    model.add(Flatten())
	    # first dropout
	    model.add(Dropout(0.25))
	    
	    # first set of FC => RELU layers
	    
	    model.add(Dense(128))
	    model.add(Activation("relu"))
	    model.add(BatchNormalization())
	    #second dropout
	    model.add(Dropout(0.5))

	    # softmax classifier
	    model.add(Dense(self.classes, activation='softmax'))
	    #model.add(Lambda(lambda x: K.argmax(x)))
	    #model.add(Reshape([-1]))
	    #model.add(Lambda(lambda x: K.cast(x,"float")))
	    
	    print(model.summary())

	    # return the constructed network architecture
	    return model
