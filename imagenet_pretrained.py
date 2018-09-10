from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
    help="name of pretrained network to use")
args = vars(ap.parse_args())

# define dict to map our models
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

if args["model"] not in MODELS.keys():
    raise AssertionError("The model does not exist")

# initialize the input image shape (224x224 pixels) along with pre-preprocessing
# function (this might need to be changed based on which model we use to
# classify our image)
inputShape = (224, 224)

preprocess = imagenet_utils.preprocess_input

# if we are using InceptionV3 or Xception networks, then we need
# to set the input shape to 299x299 and use different image preprocessing
# function
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

# load network weights from disk.
print("[INFO] loading {}...")
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# load the input image using the Keras helper util while ensuring the image
# is resized to "inputShape".
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the dimension
# by making the shape (1, inputShape[0], inputShape[1], 3) so we can pass it
# through the network.
image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the model that
# has been loaded  (i.e: mean subtraction, scaling, etc.)
image = preprocess(image)

# classify the image
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions + probabilities
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to the screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
