import sys
import os


parent = os.path.dirname(os.getcwd())
if parent not in sys.path:
    sys.path.append(parent)

root_path = parent + '/../'
if root_path not in sys.path:
    sys.path.append(root_path)

from James.CONV import leNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    print('[INFO] Fetching mnist data...')
    # dataset = datasets.fetch_mldata('mnist')
    # data = dataset.data

    # if K.image_data_format() == 'channel_first':
    #     data = np.reshape(data, (data.shape[0], 1, 28, 28))
    # else:
    #     data = np.reshape(data, (data.shape[0], 28, 28, 1))

    # (trainX, testX, trainY, testY) = train_test_split(data/255.0, data.target.astype('int'), test_size=0.25, random_state = 42)
    # le = LabelBinarizer()
    # trainY = le.fit_transform(trainY)
    # testY = le.fit_transform(testY)

    ap =argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', required=True, help='epochs to train the network')
    ap.add_argument('-m', '--model', required=True, help='file to save the trained model')
    args = vars(ap.parse_args())

    model_file = args['model']
    epochs = int(args['epochs'])

    mnist = input_data.read_data_sets('./input_data', one_hot=True, validation_size=100)
    trainX = mnist.train.images
    trainY = mnist.train.labels
    testX = mnist.test.images
    testY = mnist.test.labels
    valX = mnist.validation.images
    valY = mnist.validation.labels
    if K.image_data_format() == 'channels_last':
        trainX = np.reshape(trainX, (trainX.shape[0], 28, 28, 1) )
        testX = np.reshape(testX, (testX.shape[0], 28, 28, 1))
        valX = np.reshape(valX, (valX.shape[0], 28, 28, 1))
    else:
        trainX = np.reshape(trainX, (trainX.shape[0], 1, 28, 28) )
        testX = np.reshape(testX, (testX.shape[0], 1, 28, 28))
        valX = np.reshape(valX, (valX.shape[0], 1, 28, 28))
  

    print('[INFO]Compiling model')
    model = leNet.build(width=28, height=28, depth=1, classes=10)
    sgd = SGD(lr=0.05)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print('[INFO] Training model...')
    H = model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=32, epochs=epochs, verbose=1)

    print('[INFO] Evaluating network....')
    predicts = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predicts.argmax(axis=1), \
        target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']))
    model.save(model_file)

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epochs), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, epochs), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, epochs), H.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()





