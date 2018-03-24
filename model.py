#Load data
import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
      lines.append(line)
lines = lines[1:] #skip the first headline

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
   num_samples = len(samples)
   while 1: # Loop forever so the generator never terminates
      #shuffle(samples)
      for offset in range(0, num_samples, batch_size):
         batch_samples = samples[offset:offset+batch_size]
         images = []
         measurements = []
         for batch_sample in batch_samples:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename
            image = cv2.imread(current_path)
            b,g,r = cv2.split(image)
            image = cv2.merge([r,g,b])
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)

         # trim image to only see section with road
         X_train = np.array(images)
         y_train = np.array(measurements)
         yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)            
            
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
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data = validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model.h5')
