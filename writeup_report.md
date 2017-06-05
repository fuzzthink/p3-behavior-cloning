**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## [Rubric](https://review.udacity.com/#!/rubrics/432/view) Points

### [Template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this writeup report.

### Note on inclusion of line numbers in code
Line numbers will not be mentioned in this report if function name is referenced since it is hard to kept line numbers in sync with code changes; where as function names are a bit more stable and is easy to search.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `train.py` - create and train the model
* `datapaths.py` - image and csv filepaths
* `np_util.py`, `pd_util.py` - helper functions
* `drive.py` - Udacity provided script for driving the car in autonomous mode
* `video.py` - Udacity provided script for generating video from recorded images
* `model.h5` - model output of trained car driving convolution neural network 
* `writeup_report.md` - this file summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
simulator-path> ./linux_sim.x86_64
model-path> python drive.py model.h5
```
Simulator options selected was `640x480` and `Fastest` for "Graphics" and "Quality".

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model of my convolution neural network is based on the Nvidia Architecture featured in their [blog page](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

#### 2. Attempts to reduce overfitting in the model

The model includes ELU (Exponential Linear Unit) layers to introduce nonlinearity (`nvidia_arch`, line 84 in `train.py`). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (lines 150-155 in `train.py`). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The validation set helped determine if the model was over or under fitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 149).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (1 lap), recovering from the left and right sides of the road (~1 lap), and going slowly on sharp turns (1 lap).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nividia architecture as mentioned above. I thought this model might be appropriate because it is simple to understand and is shown to produce good results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

The final architecture produced the following losses:

    + Epoch 1: .0326 loss; .0080 validation loss
    + Epoch 2: .0081 loss; .0074 validation loss
    + Epoch 3: .0148 loss; .0079 validation loss

Epoch 3 seems to be getting worst so it indicates possible overfitting.
The final model is a run with 2 epochs. Although running it with 3 epochs seems no difference.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I tried gathering better training data. Trying the Vgg16 architecture. Due to the slow GPU I have, training takes hours even for just a few epochs. So it is difficult to experiment with too many different parameters and layers.

In the end, the vehicle is able to drive autonomously around the track without leaving the road.  See Results section for more details.

#### 2. Final Model Architecture

The final model architecture (`train.py` lines 84-95) consisted of the Nvidia convolution neural network with the following layers and layer sizes:

  - 1 Input layer of 64 x 64 x 3 image channels
  - Conv2D layer of 24 5x5 kernels; 2x2 stride; ELU activation;
  - Conv2D layer of 36 5x5 kernels; 2x2 stride; ELU activation;
  - Conv2D layer of 48 5x5 kernels; 2x2 stride; ELU activation;
  - Conv2D layer of 64 3x3 kernels; 1x1 stride; ELU activation;
  - Conv2D layer of 64 3x3 kernels; 1x1 stride; ELU activation;
  - 1 flattened dim of convolutional layers;
  - 1 fully connected Dense layer; 1000 output dims;
  - 1 fully connected Dense layer; 100 output dims;
  - 1 fully connected Dense layer; 50 output dims;
  - 1 fully connected Dense layer; 10 output dims;
  - 1 fully connected Dense layer; 1 output dim; Linear activation;

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving under medium speed overall.

I then run another lap running full speed on straight road, but slow down on tough turns so that more images and data points can be captured on the trouble turns.

These tow laps make up about 6000 records.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it is heading off road for both straight roads and sharp turns.

These make up about 3000 records.

These are combined together giving a total of over 9200 records.

The first step is to remove those of speed less than 1 mph.

This gives a total record count of 9075.

Next the number of records are doubled by copying the records but indicating that the center image will receive a mirror operation later in the pipeline and changing its steering value to its negation.

This gives a total record count of 18,150.

Next the data is randomly shuffled.

These steps are done in the middle of `train.py`:
```python
log = pdu.filter_gte(log, 'speed', 1)
mlog = pdu.mirror(log, 'center', 'steering')
mlog = pdu.shuffle(mlog)
```

This data set is split into 80% training and 20% validation.

This gives a training data set of 14520 records, and 3630 in validation.

The training set and validation set is generated with a batch generation function (`getBatchGenFn()` in `train.py` generates this function). An image processor function (`getCropResizeFn()` in `np_util.py` generates this function) is passed to it that is defined to do the following processing steps:

1. Crop the image at top:69, bottom:27, left:10, right:10

Since the images are 320 x 160, the resulting image is 300 x 64.

2. Resize to 64 x 64. Since the height is already 64 pixels, only scaling is in the horizontal direction.

3. Convert to BGR and normalize the pixels.

4. Flip the image if is was marked to be flipped above.

#### 4. Results

Adding more epochs to training did not seem to help getting the car to steer correctly at sharp turns reliably. Usually within a few laps, it will steer into a sharp turn the wrong way (too straight) and can not recover. What really help was changing the throttle such that it does not go on full speed.

Here is a summary of what works and failure rates:

Constant throttle:
| throttle | max speed | off course in | 
| :---: | :---: | :---: |
| .3 and up | 30 mph | 1 or 2 laps |
| .2 | 23 mph | 3 laps |
| .15 | 17 mph | 5 laps |
| .1 | 11 mph | Did not go off course in over 1.5 hrs, ~50 laps |

So .1 throttle is good. But thought I can do better.

I up the throttle at .2 and when the absolute value of steering input is over 4, I brake at .0001 x absolute steering value. This slows it down to ~11 mph at sharp turns. Also, since the model is not compensating for the steering enough at sharp turns, I multiply the steering by 2 if absolute steering input is over 4 and multiply it by 2.5 if over 6.

This setting runs the laps at a much higher speed and it ran over 1.5 hrs, or over 66 laps without going off course.

#### 5. Thoughts

This project is both very enjoying and frustration. It is fun to see the model run forever without going off coure at the end. But with a slow GPU, the feedback process is poor. I will either invest in new GPU or just cloud offerings to continue trying different the model architecture, layers, and parameters to get it to run full speed and to complete the challenge course as well. 

