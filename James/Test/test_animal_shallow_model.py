import sys
import os

parent_path = os.path.dirname(os.getcwd())
if parent_path not in sys.path:
    sys.path.append(parent_path)

root_path = parent_path + '/../'
if root_path not in sys.path:
    sys.path.append(root_path)

from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
    ap.add_argument('-m', '--model', required=True, help='file to load the trained model')
    args = vars(ap.parse_args())

    classLabels = ['cat', 'dog', 'panda']

    print("[INFO] sampling images...")
    imagePaths = np.array(list(paths.list_images(args["dataset"])))
    idxs = np.random.randint(0, len(imagePaths), size=(10,))
    imagePaths = imagePaths[idxs]

    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths)

    data = data.astype('float')/255.0

    model = load_model(args['model'])

    print('[INFO] predicting...')
    predicts = model.predict(data, batch_size=32).argmax(axis=1)

    for (i, path) in enumerate(imagePaths):
        img = cv2.imread(path)
        cv2.putText(img, "Label {}".format(classLabels[predicts[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
        cv2.imshow('Image', img)
        cv2.waitKey(0)


