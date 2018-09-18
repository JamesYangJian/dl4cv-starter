import sys
import os

parent_path = os.path.dirname(os.getcwd())
if parent_path not in sys.path:
    sys.path.append(parent_path)

root_path = parent_path + '/../'
if root_path not in sys.path:
    sys.path.append(root_path)

from keras.models import load_model
import argparse
import numpy as np
import cv2
import threading
import time

class imagePanel:
    def __init__(self):
        self.img = np.zeros((320, 320), dtype = np.uint8)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.point_pic)
        self.predict = 0
        self.drawing = False

    def point_pic(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing:
                self.drawing = False
            else:
                self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.img, (x, y), 5, (255), -1)

    def show_pic(self):
        cv2.imshow('img', self.img)

    def stop_drawing(self):
        self.drawing = False

    def clear_img(self):
        self.img = np.zeros((320, 320), dtype = np.uint8)
        self.stop_drawing()

    def put_text(self, text):
        cv2.putText(self.img, "Number {}".format(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help='file to save the trained model')
    args = vars(ap.parse_args())

    path = '/Users/yangjian/work/opencv/OpencvTutorials-master/output/'

    model = load_model(args['model'])

    panel = imagePanel()

    # picThd = draw_pic_thread(panel)
    # picThd.start()
 

    while True:
        panel.show_pic()
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('c'):
            panel.clear_img()
        elif key == ord('p'):
            panel.stop_drawing()

            predict_img = cv2.resize(panel.img, (28, 28))
            cv2.imshow('number', predict_img)
            # predict_img = cv2.cvtColor(predict_img, cv2.COLOR_BGR2GRAY)
            predict_img = np.reshape(predict_img, (1, 28, 28, 1))
            predict = model.predict(predict_img).argmax()

            print('This image is %s' % str(predict))
            panel.put_text(str(predict))


        # file_name = input('Please input image file name:')
        # if file_name == 'exit':
        #     break

        # file_full_name = path + file_name
        # try:
        #     img = cv2.imread(file_full_name)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     img = np.reshape(img, (1, 28,28,1))
        #     predict = model.predict(img).argmax()
        #     print('The predict value is %s' % str(predict))
        # except Exception as e:
        #     print('Error: %s' % str(e))
