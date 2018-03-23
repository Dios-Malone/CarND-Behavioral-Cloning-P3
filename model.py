#Load data
import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
      lines.append(line)
      
images = []
measurements = []
lines = lines[1:] #skip the first headline
for line in lines:
   source_path = line[0]
   filename = source_path.split('/')[-1]
   current_path = 'data/IMG/' + filename
   image = cv2.imread(current_path)
   b,g,r = cv2.split(image)
   image = cv2.merge([r,g,b])
   images.append(image)
   measurement = float(line[3])
   measurements.append(measurement)
   
X_train = np.array(images)
y_train = np.array(measurements)


#Model Architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x - 127.0)/255.0, input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 2, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 2, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 2, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 5, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 5, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')