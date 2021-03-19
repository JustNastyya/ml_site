import numpy as np
import random
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from PIL import Image


def predict(image, model):
    im = Image.open(image)
    im = im.convert('L').resize((28, 28))
    im = np.array(im, dtype='float32')
    im = np.reshape(im, (1, 784))
    im = im / 255

    prediction = model.predict_classes(im)[0]

    print(f'prediction on image {image}:', prediction)



model = load_model('digits_rec_model.h5')
images = ['five.png', 'nine.png', 'one_diff.png', 'one.png', 'seven.png', 'six.png', 'zero.png']
path = 'images\\'

for im in images:
    predict(path + im, model)