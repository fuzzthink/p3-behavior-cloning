import numpy as np
import cv2
from PIL import Image
from keras.preprocessing import image as kImg
from keras.applications import imagenet_utils

def scale_255(M):
    return np.uint8(255*M/np.max(M))

def crop(img, top=0, btm=0, left=0, right=0):
    if hasattr(img, 'crop') and callable(getattr(img, 'crop')):
        # PIL image
        wd,ht = img.size[0], img.size[1]
        return img.crop((left, top, wd-10, ht-10))

    elif hasattr(img, 'shape'):
        # numpy array, eg. via cv2.imread()
        ht,wd,_ = img.shape
        return img[top:ht-btm, left:wd-right]

    else:
        raise ValueError('img type '+type(img)+' not expect')

def resize(img, wd_or_shape, ht=None):
    shape = wd_or_shape if ht==None else (wd_or_shape, ht)
    if hasattr(img, 'resize') and callable(getattr(img, 'resize')):
        # PIL image
        return img.resize(shape, Image.ANTIALIAS)

    elif hasattr(img, 'shape'):
        # numpy array, eg. via cv2.imread()
        return cv2.resize(img, shape)

    else:
        raise ValueError('img type '+type(img)+' not expect')

def getCropResizeFn(top=0, btm=0, left=0, right=0, shape=(64,64)):
    ''' Returns a function that crop, resize, convert to BGR, normalize pixels,
        and L<->R flips the image(if specified).
        top, btm, left, right: params for crop
        shape: shape to resize to 
    '''
    def fn(imgpathio):
        ''' load image from imgpathio (filepath or io)
        If filepath ends with '_mirror', it indicates the image should be 
         flipped L<->R. But the real filepath is without the '_mirror'
        '''
        if type(imgpathio)==str and imgpathio.endswith('_mirror'):
            imgpathio = imgpathio[:-7] # remove ending '_mirror'
            isMirror = True
        else:
            isMirror = False

        img = kImg.load_img(imgpathio) # keras load_img uses PIL
        img = crop(img, top, btm, left, right)
        img = resize(img, shape)

        # imagenet_utils.preprocess_input takes a batch of images, converts to
        #  BGR and normalize pixels by subtracting mean pixel
        #  https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
        # np.expand_dims() turns image into a tensor/batch 
        imgBatch = np.expand_dims(kImg.img_to_array(img), axis=0)
        x = imagenet_utils.preprocess_input(imgBatch)[0]
        if isMirror:
            x = x[:,::-1,:] # L<->R flip
        return x
    return fn