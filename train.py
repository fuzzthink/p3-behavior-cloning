import numpy as np
import pandas as pd
import np_util as npu
import pd_util as pdu
import cv2

from os import path
from datapaths import datapaths, testpaths
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D


def getBatchGenFn(df, batch_size, xTrainCol, yTrainCol, imgFn, altTrainCols=[],
                  altTrainAdj=[], bothAltsPerSample=False):
    ''' Returns a batch generator function that yields a batch of [xTrains] to
        corresponding [yTrains] 
    df:    pandas dataframe
    xTrainCol, yTrainCol: col names in df
    imgFn:        function to apply to xTrain
    altTrainCols: list of col names (up to 2) to add to batch addition to xTrainCol
                  No additions are added if list is []
    altTrainAdj:  list of amounts (up to 2) of adjustments to yTrain corresponding
                  to col in altTrainCols
    bothAltsPerSample: add both col adjustments per record if true. Otherwise,
                  only 1 per record (alternates btw the 2) is added.
    '''
    def _generator(df):
      batch = []
      iAltTrain = 0
      while True:
        for row in df.itertuples():
            xtrain = imgFn(row[colIdx[xTrainCol]])
            steer = row[colIdx[yTrainCol]]
            batch.append([xtrain, steer])

            if altTrainCols:
                col = altTrainCols[iAltTrain]
                xtrain = imgFn(row[colIdx[col]])
                steer0 = steer + altTrainAdj[iAltTrain]
                iAltTrain = (iAltTrain + 1)%2
                batch.append([xtrain, steer0])

                if bothAltsPerSample:
                    col = altTrainCols[iAltTrain]
                    xtrain = imgFn(row[colIdx[col]])
                    steer1 = steer + altTrainAdj[iAltTrain]
                    iAltTrain = (iAltTrain + 1)%2
                    batch.append([xtrain, steer1])

            if len(batch) >= batch_size:
                current_batch = batch[:batch_size]
                yield tuple(map(np.asarray, zip(*current_batch)))
                batch = batch[batch_size:]
    return _generator(df)


img_cols_str = 'center left right'
headers = (img_cols_str+' steering throttle break speed').split(' ')
colIdx = {k:v+1 for v,k in enumerate(headers)} # +1 since col 0 is the row index

def imgs_log_gen(datapaths, headers, img_cols_str):
    ''' Returns pandas log of image filenames
    '''
    img_cols = img_cols_str.split(' ')
    logs = []

    for csvpath, imgspath in datapaths:
        log = pd.read_csv(csvpath, header=None, names=headers)
        for col in img_cols:
            log[col] = log[col].str.rsplit('/', n=1).str[-1].apply(
                lambda s: path.join(imgspath, s)
            )
        logs.append(log)
    return pd.concat(logs, axis=0, ignore_index=True)

log = imgs_log_gen(datapaths, headers, img_cols_str)

## Filter out records of speed < 1; Add mirror col for center image; shuffle. 
log = pdu.filter_gte(log, 'speed', 1)
mlog = pdu.mirror(log, 'center', 'steering')
mlog = pdu.shuffle(mlog)

nvidia_arch = [
    Conv2D(24,5,5, subsample=(2,2), activation='elu', input_shape=(64,64,3)),
    Conv2D(36,5,5, subsample=(2,2), activation='elu'),
    Conv2D(48,5,5, subsample=(2,2), activation='elu'),
    Conv2D(64,3,3, subsample=(1,1), activation='elu'),
    Conv2D(64,3,3, subsample=(1,1), activation='elu'),
    Flatten(),
    Dense(1000),
    Dense(100),
    Dense(50),
    Dense(10),
    Dense(1, activation='linear'),
]    

from keras.applications import VGG16
from keras.layers import AveragePooling2D, Dropout, BatchNormalization, Input
from keras.regularizers import l2

def vgg16_model():
    img = Input(shape=(64,64,3))
    basemodel = VGG16(input_tensor=img, include_top=False)
    for layer in basemodel.layers[:-3]:
        layer.trainable = False
    x = basemodel.get_layer('block5_conv3').output
    x = AveragePooling2D((2,2))(x) 
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='elu', W_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='elu', W_regularizer=l2(0.01))(x)
    x = Dense(2048, activation='elu', W_regularizer=l2(0.01))(x)
    x = Dense(1, activation='linear')(x)
    return Model(input=img, output=x)

## Generate model
model = Sequential(nvidia_arch)
# model = vgg16_model()
model.summary()

## Generate train and validation set
train_set, val_set = pdu.train_test_split(mlog, .2)
train_size = train_set.shape[0]
val_size = val_set.shape[0]

print('data size before mirror', log.shape[0])
print('data size after mirror', mlog.shape[0])
print('train size', train_size)
print('val size', val_size)

batch_size = 128
epochs = 3
cols = ['center', 'steering']
LRcols=['left', 'right']
processCols = ['center']

imgFn = npu.getCropResizeFn(69, 27, 10, 10, (64,64))
adjust = 2.5 #steering adjustment to make L/R images usable for training.
# failed: .7, 1.5, 2,  2.8, 3.3
adjLR = [adjust, -adjust]

train_generator=getBatchGenFn(train_set,batch_size,'center','steering',imgFn,LRcols,adjLR)
val_generator = getBatchGenFn(val_set, batch_size, 'center','steering',imgFn)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator, 
    samples_per_epoch=train_size,
    validation_data=val_generator, 
    validation_steps=val_size,
    epochs=epochs,
)
model.save('model.h5')
