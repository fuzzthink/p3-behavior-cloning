
To train a computer to drive itself, images from a camera attached to the car and its corresponding steering and other outputs can be feed to a convolutional neural network to learn its own outputs when it encounters similar data. This technique is called behavorial cloning.

The goals / steps of this [project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

---
### Environment Setup

The environment can be created with [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The Udacity Unity Car Simulator can be downloaded [here](https://github.com/udacity/self-driving-car-sim). Ignore the instructions to install Unity, as Unity loads automatically with executable. For Linux, you may need to `chmod 777 linux_sim.x86_64` it if they still have not done that already.


---
### Source Files 
The project includes the following files:

- `train.py` - train the convolution neural network model and save it
- `datapaths.py` - defines image and csv file paths
- `np_util.py`, `pd_util.py` - helper functions
- `drive.py` - Udacity provided script for driving the car in autonomous mode
- `video.py` - Udacity provided script for generating video from recorded images

---
### Training 
The model is trained by a dataset of images from the car camera's point of view of the road and its corresponding steering angles. These are specified in `datapaths.py`. The recorded dataset used can be found in [here](https://github.com/fuzzthink/P3-recorded-data).

You can train your model with above recorded dataset, use [Udacity's recorded dataset](http://d2uz2655q5g6b2.cloudfront.net/46a70500-493e-4057-a78e-b3075933709d/169019/Behavioral%20Cloning%20Videos.zip), or generate your own data via the record feature in the simulator. To do so, first start the simulator.
```sh 
simulator-path> ./linux_sim.x86_64
```

Selected `640x480` for "Graphics" and `Fastest` for "Quality".

Select "Training Mode" and click the record button when you like to start recording. It will ask you where the video images will be saved to. You will need to modify the paths in `datapaths.py` to where the images and csv is saved to.

Once you have recorded at least one lap of images, you can start the training via:
```sh 
python train.py
```

This will save the model to `model.h5` when complete. Training takes a few hours depending on the GPU you use.


---
### Running the model
When training is complete, you can see it in action by selecting "Autonomous Mode" in the simulator and run the model via:
```sh 
python drive.py model.h5
```

To produce a video recording of the run from the car camera's point of view, hit the record button in the simulator and stop it when you like to stop. Then run:
```sh 
python video.py RECORDED_IMAGES_PATH
```
This will create a mp4 video output of car's point of view of the run.

---
### Model Architecture and Training Strategy

The model of my convolution neural network is based on the Nvidia Architecture featured in their [blog page](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) or their [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. It uses an adam optimizer and includes ELU (Exponential Linear Unit) layers to introduce nonlinearity. The trained model is able to drive autonomously around the track without leaving the road after some modification to `drive.py` (more on this in Results section below).

A combination of center lane driving (1 lap), recovering from the left and right sides of the road (~1 lap), and going slowly on sharp turns (1 lap).

The training data is split into a training and validation set. This helps to gauge how well the model is working and to determine if the model was over or under fitting.

The final architecture produced the following losses:

    + Epoch 1: .0326 loss; .0080 validation loss
    + Epoch 2: .0081 loss; .0074 validation loss
    + Epoch 3: .0148 loss; .0079 validation loss

Epoch 3 seems to be getting worst so it indicates possible overfitting.

The final model is a run with 2 epochs. Although running it with 3 epochs seems no difference.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I tried gathering better training data and also tried training with the Vgg16 architecture. Since training takes hours even for just 2 - 3 epochs, it is difficult to experiment with too many different parameters and layers.

The final model architecture consists of the Nvidia convolution neural network with the following layers and layer sizes:

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

---
### Creation of the Training Set & Training Process

To capture good driving behavior, one lap on track 1 using center lane driving under medium to medium-high speed is recorded.

I then run another lap running full speed on straight roads, but slow down on tough turns so that more images and data points can be captured on the trouble turns.

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

---
### Results

Adding more epochs to training did not seem to help getting the car to steer correctly at sharp turns reliably. Usually within a few laps, it will steer into a sharp turn the wrong way (too straight) and can not recover. After not getting any better results from completely different training recordings and more tough turns recordings, and trying different parameters and layers, I had to think a bit outside the box. Instead of getting a better model to make these turns better, what if slow down and not run the simulation at full throttle?
To do so, I modify `drive.py` with the following throttle settings:

Here is a summary of what throttle settings work and its failure rates:

| throttle | max speed | went off course in | 
| :---: | :---: | :---: |
| 0.3 and up | 30 mph | 1 or 2 laps |
| 0.2 | 23 mph | 3 laps |
| 0.15 | 17 mph | 5 laps |
| 0.1 | 11 mph | Did not go off course in over 1.5 hrs, ~50 laps |

So throttle of 0.1 is good. But thought I can do better.

What if I amplify the model by upping the speed at straight roads and slowing down and steering more than model suggests at sharp turns? 

I set the throttle at .2 nornally and when the absolute value of steering input is over 4, I brake at .0001 x absolute steering value. This slows it down to ~11 mph at sharp turns. The steering is multiplied by 1.5 if absolute steering input is over 4 and multiplied by 1.8 if over 6.

This setting runs the laps at a much higher speed and it ran over 1.5 hrs, or over 66 laps without going off course.

---
### Thoughts

I have a love and hate relationship with this project. It is fun to see the model run forever without going off course at the end. But with a slow GPU, the feedback process is poor. I want to see the model work on the very difficult track two. To do so, I will need to either invest in new GPU or use cloud offerings to continue trying different the model architecture, layers, and parameters. 

---
##### [Project Review](UdacityReviews.pdf)
