import numpy as np
import random
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import np_utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# https://dmitryulyanov.github.io/deep_image_prior

width, height = (1024, 1024)
AMOUNT_OF_DATA = 2# 202599


def image_to_nparray():
    i = 1
    x_train = np.empty((1, width, height, 3))
    
    name = (6 - len(str(i))) * '0' + str(i) + '.jpg'
    name = 'datasets\\dataset\\' + name
    im = Image.open(name)
    data = np.asarray(im)
    print(data.shape)
    x_train[0] = np.asarray(im)
    return x_train


def prepare_data(AMOUNT_OF_DATA):
    order = ['t' + str(i) for i in range(AMOUNT_OF_DATA)]
    for i in range(AMOUNT_OF_DATA):
        order.append('f' + str(i))
    random.shuffle(order)

    len_train_set = int(0.6 * (AMOUNT_OF_DATA * 2 + 1))
    AMOUNT_OF_DATA = AMOUNT_OF_DATA * 2
    print('len_train_set:', len_train_set)
    x_train = np.empty((len_train_set, width // 2, height // 2, 3), dtype=np.uint8)
    y_train = np.empty((len_train_set, 1), dtype=np.uint8)
    x_test = np.empty((AMOUNT_OF_DATA - len_train_set, width // 2, height // 2, 3), dtype=np.uint8)
    y_test = np.empty((AMOUNT_OF_DATA - len_train_set, 1), dtype=np.uint8)
    
    for i in range(len_train_set):
        name = (6 - len(order[i])) * '0' + str(order[i][1:]) + '.png'
        if order[i][0] == 't':
            im = Image.open('datasets\\dataset_real\\' + name).resize((width // 2, height // 2))
            x_train[i] = np.array(im)
            y_train[i] = 1
        else:
            try:
                im = Image.open('datasets\\dataset_unreal\\' + name).resize((width // 2, height // 2))
            except Exception:
                try:
                    im = Image.open('datasets\\dataset_unreal\\0' + name).resize((width // 2, height // 2))
                except Exception:
                    pass
            x_train[i] = np.array(im)
            y_train[i] = 0

    for i in range(len_train_set, len(order)):
        name = (6 - len(order[i])) * '0' + str(order[i][1:]) + '.png'
        if order[i][0] == 't':
            im = Image.open('datasets\\dataset_real\\' + name).resize((width // 2, height // 2))
            x_test[i - len_train_set] = np.array(im)
            y_test[i - len_train_set] = 1
        else:
            try:
                im = Image.open('datasets\\dataset_unreal\\' + name).resize((width // 2, height // 2))
            except Exception:
                try:
                    im = Image.open('datasets\\dataset_unreal\\0' + name).resize((width // 2, height // 2))
                except Exception:
                    pass
            x_test[i - len_train_set] = np.array(im)
            y_test[i - len_train_set] = 0

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


def check_if_dataset_is_built_correctly():
    x_train, y_train, x_test, y_test = prepare_data(10)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    for i in range(x_train.shape[0]):
        im = Image.fromarray(x_train[i])
        im.save('datasets\\datasets_check\\x_train\\' + str(i) + ".jpg")
        #im = Image.fromarray(y_train[i])
        #im.save('datasets\\datasets_check\\y_train\\' + str(i) + ".jpg")

    for i in range(x_test.shape[0]):
        im = Image.fromarray(x_test[i])
        im.save('datasets\\datasets_check\\x_test\\' + str(i) + ".jpg")
        #im = Image.fromarray(y_test[i])
        #im.save('datasets\\datasets_check\\y_test\\' + str(i) + ".jpg")


def model_built(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width // 2, height // 2, 3), strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def model_train(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save('models\\' + str(score) + '_model.h5')
    return score, history


def evaluate_model(history, model, x_test, y_test):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def main():
    x_train, y_train, x_test, y_test = prepare_data(50)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    
    model = model_built(x_train, y_train, x_test, y_test)
    score, history = model_train(model, x_train, y_train, x_test, y_test)
    print()
    print('score: ', score)
    print('starting validation')
    evaluate_model(history, model, x_test, y_test)


if __name__ == '__main__':
    main()
