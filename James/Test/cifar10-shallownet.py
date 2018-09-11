from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import os
import sys

parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

root_dir = parent_dir + '/../'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from CONV import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    print('[INFO] loading cifar10 data')
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype('float')/255.0
    testX = testX.astype('float')/255.0

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print('[INFO] compiling model....')
    opt = SGD(lr=0.01)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print('[INFO] training network...')
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=40, verbose=1)

    print('[INFO] evaluating model...')
    prediction = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), prediction.argmax(axis=1), target_names=labelNames))

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 40), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, 40), H.history['val_acc'], label='val_acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()