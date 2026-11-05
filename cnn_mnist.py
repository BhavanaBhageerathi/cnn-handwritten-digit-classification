
'''
HAND WRITTEN DIGIT CLASSIFICATION USING CNN-MNIST DATASET
'''
"""
1)load and fetch data
"""

#THE STANDARD MNIST DATASET IS INCLUDED IN KERAS LIBRARY ITSELF SO WE JUST HAVE TO IMPORT IT.
#it is already cleaned and test and train data is already split
#all images are numpy arrays(always nammal images numpy aray ayit convert akanam for recognition)
from keras.datasets import mnist
import matplotlib.pyplot as plt

#it is already cleaned and test and train data is already split just have to load it into seperate tuple variables
(x_train,y_train),(x_test,y_test)=mnist.load_data()

"""2)EDA"""

#try to display some images
plt.imshow(x_train[1],cmap=plt.get_cmap('gray'))

x_train.shape
#the current shape is  (60000, 28, 28)
#we need to include channel data also
#we need to convert that into(60000, 28, 28,1)

"""3)DATA PREPROCESSING"""

import numpy as np
from keras.utils import to_categorical#for one-hot-encoding

#Data preprocessing -step1
#reshaping the input data into[no_of_rows][width][height][channels] channels means depth grey-1 colour-3
#to include channel data also
x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')# or x_train=x_train.reshape(6000,28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

print(x_train.shape)
print(x_test.shape)

#Data preprocessing -step2
#normalise the pixel range from 0 to 255 to 0 to 1
x_train=x_train/255 #by dividing it by 255
x_test=x_test/255

#Data preprocessing -step3
#one hot encod the output data(last kitenda classes ahn..so x data alla y data ahn encod chyendath)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#define num of classes in  variable
num_classes=y_test.shape[1]

"""4)MODEL DEFINING"""

#import required classes to create the layers of cnn
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
#Conv2D-convolutional layer,MaxPooling2D-pooling layer,Flatten-flattten layer,Dense-fully connected layer,Dropout

#define the layers of cnn
#create the entire model as a function
def mymnistmodel():
  #create a sequentional model
  mymodel=Sequential()

  ############### DEEP LEARNING CONVOLUTIONAL NETWORK #################
  #add the conv layer
  #(Conv2D(num of feature map,(each map size),padding,activation funct,shape
  mymodel.add(Conv2D(32,(5,5),padding='valid',activation='relu',input_shape=(28,28,1)))

  #add a max pooling layer with a pooling size of 2x2
  mymodel.add(MaxPooling2D(pool_size=(2,2)))

  #add a dropout layer with a dropout rate of 0.2#random ayit 20% off avum to reduce overfitting
  mymodel.add(Dropout(0.2))

  #add a flatten layer to flatten the convolution output so that we can pass that as input to a normal multi layer percepron network becuz only mlp can make predictions
  mymodel.add(Flatten())

  ######### NORMAL DEEP LEARNING MLP NETWORK #################

#add a fully connected layer for our MLP input
  mymodel.add(Dense(128,activation='relu')) #why 128 0-255 ayond athinte oru half edthenae olu no such reason

#add an outputlayer with 10 nodes with softmax activation funct
  mymodel.add(Dense(num_classes,activation='softmax'))

#compile the model
  mymodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  return mymodel

#get the model by calling the function
mymnistmodel = mymnistmodel()

"""5)MODEL FITTING"""

#fit the data into the model and run traing
mymnistmodel.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=200,verbose=2)

"""6)MODEL EVALUATION"""

mymnistmodel.evaluate(x_test,y_test)

#we can save this model and load this when we need it this process is called serializing the model and saving it
#when we save,we need to save 2 things..model architecture and weight

#1)save the structure of model
path="/content/drive/MyDrive/faith_csvfiles/model_mnist.json"
mymnistmodel_schema_json = mymnistmodel.to_json()
#writing data
with open(path, "w") as json_file:  # file management:open il write modeil ah
    json_file.write(mymnistmodel_schema_json)

path="/content/drive/MyDrive/faith_csvfiles/model_mnist.weights.h5"
#2)serialize/save the model weights a HDF5 file (with.h5 extension)
mymnistmodel.save_weights(path)

"""LOAD THE TRAINED MODEL FROM SAVED FILES"""

from keras.models import model_from_json
from PIL  import Image #nammal ivde image ahn kodukune input ayit sao ee lib use akanam
import numpy

#load the json model structure and create the model

#read the json
path="/content/drive/MyDrive/faith_csvfiles/model_mnist.json"
json_file = open(path, "r") #avde write mode uvde read mode
mymodel_json=json_file.read()
json_file.close()

#create model from json
loaded_mymodel=model_from_json(mymodel_json)

#now load the weights into the model
path="/content/drive/MyDrive/faith_csvfiles/model_mnist.weights.h5"
loaded_mymodel.load_weights(path)

loaded_mymodel.summary()

#load the image to predict
imgpath="/content/drive/MyDrive/faith_csvfiles/7.png"
myimg= Image.open(imgpath).convert('L')#to convert it into greyscale

#resize the image to a 28x28 image
myimg= myimg.resize((28,28))

#convert the resized image into a numpy array
myimgarr = np.array(myimg)

#reshape the numpy array to include the dimension and channel value
myimgarr = myimgarr.reshape(1,28,28,1)

#trying the prediction
predicted_array = loaded_mymodel.predict(myimgarr)

#get the most probability item from the array
print(predicted_array.argmax())