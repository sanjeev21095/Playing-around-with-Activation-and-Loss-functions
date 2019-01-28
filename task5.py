import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from scipy.misc import imread
#importing the activation and loss function libraries
from custom_activations import cust_act
from keras.custom_losses import cubed_hinge_loss 
K.set_image_dim_ordering('th')

####################3
#For randomness
seed = 7
numpy.random.seed(seed)
 
# Uploading mnist data to the program
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to match the mnist image convention
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
#Normalizing the dataset
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Fix the activation function
act1 = cust_act()

#defining the model
def larger_model():
	model = Sequential()
    #Insert activation as class and not a label
	model.add(Convolution2D(96, (11, 11), padding='valid', input_shape=(1, 28, 28), activation=act1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(48, (5, 5), activation=act1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(256, activation=act1))
	model.add(Dense(128, activation=act1))
	model.add(Dense(num_classes, activation=act1))
    #Insert loss as a class and not a label
	model.compile(loss=cubed_hinge_loss, optimizer='adam', metrics=['accuracy'])
	return model
#Display training values
model = larger_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#***************************************************************
#Test with input image
# load an image
image = imread('1.png').astype(float)
image = image / 255.0
image = image.reshape(1,1,28,28)
    
print("predicted digit: "+str(model.predict_classes(image)[0]))


