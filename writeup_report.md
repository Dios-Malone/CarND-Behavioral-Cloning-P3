# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 is a recording of simulation in autonomous model

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 64-88) 

The model includes RELU layers to introduce nonlinearity (code line 69, 72, 75, 78, 81), and the data is normalized in the model using a Keras lambda layer (code line 67). 

#### 2. Attempts to reduce overfitting in the model

The model contains pooling layers in order to reduce overfitting (model.py lines 70, 73, 76, 79, 82). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 93). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 90).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only used the center view images from the provided sample data. And it works very well.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the model architecture described in the paper (Title: End to End Learning for Self-Driving Cars) from NVIDIA Corporation. I thought this model might be appropriate because the model architecture was designed to achieve a similar target but on a real car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that max pooling was done between each convolution layer.

The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicle fell off the track. It was a place having a mud side-road. To improve the driving behavior in this case, I recorded some more training data at this place and continue training the model with the additional data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-88) consisted of a convolution neural network with the following layers and layer sizes:
1. Normalization Layer
2. Convolution layer with 5x5 kernel and 2x2 strides and 24 filters
3. RELU Activation Layer
4. Max Pooling Layer with 2x2 pool size, no stride and valid padding
5. Convolution layer with 5x5 kernel and 2x2 strides and 36 filters
6. RELU Activation Layer
7. Max Pooling Layer with 2x2 pool size, no stride and valid padding
8. Convolution layer with 5x5 kernel and 2x2 strides and 48 filters
9. RELU Activation Layer
10. Max Pooling Layer with 2x2 pool size, no stride and valid padding
11. Convolution layer with 5x5 kernel and 3x3 strides and 64 filters
12. RELU Activation Layer
13. Max Pooling Layer with 2x2 pool size, no stride and valid padding
14. Convolution layer with 5x5 kernel and 3x3 strides and 64 filters
15. RELU Activation Layer
16. Max Pooling Layer with 2x2 pool size, no stride and valid padding
17. Flantten Layer
18. Fully connected layer with size 100
19. Fully connected layer with size 50
20. Fully connected layer with size 1


#### 3. Creation of the Training Set & Training Process

I first used the provided sample data and found the result surprisingly good. In this case, I didn't record many training data except some places where the car driven off the track. 

I captured some more data at the places where the pre-trained model failed and using them to continue training the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by experiment. I used an adam optimizer so that manually training the learning rate wasn't necessary.
