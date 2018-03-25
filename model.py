#Load data
import csv
import cv2
import numpy as np

#=====Reading training samples from file=====
lines = []
with open('data/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
      lines.append(line)
lines = lines[1:] #skip the first headline


#=====Shuffle and split samples=====
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
lines = shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


#=====Implemeting generator for input data generation=====
def generator(samples, batch_size=32):
   num_samples = len(samples)
   while 1: # Loop forever so the generator never terminates
      for offset in range(0, num_samples, batch_size):
         batch_samples = samples[offset:offset+batch_size]
         images = []
         measurements = []
         for batch_sample in batch_samples:
            source_path = batch_sample[0]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename
            #Read image data
            image = cv2.imread(current_path)
            #Convert pixel data from BGR to RGB
            b,g,r = cv2.split(image)
            image = cv2.merge([r,g,b])
            #Add image data to generator output
            images.append(image)
            #Add label data to generator output
            measurement = float(batch_sample[3])
            measurements.append(measurement)
         X_train = np.array(images)
         y_train = np.array(measurements)
         yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)            

            
#=====Model Architecture=====
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import os.path

if os.path.isfile('model.h5'): 
   # if a trained model exists, continue training based on the pre-trained model
   model = load_model('model.h5')
else:
   # else create a new model and train it from scratch 
   model = Sequential()
   #Normalize layer
   model.add(Lambda(lambda x: (x - 127.0)/255.0, input_shape=(160,320,3)))
   #1st Convolution layer with 5x5 kernel and 2x2 strides
   model.add(Convolution2D(24, 5, 2, activation="relu"))
   model.add(MaxPooling2D())
   #2nd Convolution layer with 5x5 kernel and 2x2 strides
   model.add(Convolution2D(36, 5, 2, activation="relu"))
   model.add(MaxPooling2D())
   #3rd Convolution layer with 5x5 kernel and 2x2 strides
   model.add(Convolution2D(48, 5, 2, activation="relu"))
   model.add(MaxPooling2D())
   #4th Convolution layer with 5x5 kernel and 3x3 strides
   model.add(Convolution2D(64, 5, 3, activation="relu"))
   model.add(MaxPooling2D())
   #5th Convolution layer with 5x5 kernel and 3x3 strides
   model.add(Convolution2D(64, 5, 3, activation="relu"))
   model.add(MaxPooling2D())
   #Flatten layer
   model.add(Flatten())
   #Three Fully connected layers
   model.add(Dense(100))
   model.add(Dense(50))
   model.add(Dense(1))
   #Using MSE loss function and AdamOptimizer
   model.compile(loss='mse', optimizer='adam')

#Train model   
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data = validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

#Save model
model.save('model.h5')
