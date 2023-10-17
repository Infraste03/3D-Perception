# These lines of code are importing the necessary libraries for the script to run:
import cv2
import argparse
import numpy as np


# The code block is using the `argparse` module to define and parse command-line arguments.
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
help = 'path to text file containing class names')
args = ap.parse_args()


image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
# The line `scale = 0.00392` is setting the scale factor for the image. This scale factor is used to
# normalize the pixel values of the image before passing it through the neural network. In this case,
# the scale factor of 0.00392 is used to divide the pixel values by 255, which is the maximum value
# for a pixel in an 8-bit image. This normalization is necessary because the pre-trained YOLO model
# expects input images to be in the range of 0 to 1.
scale = (1/255)

# read class names from text file
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
# `net = cv2.dnn.readNet(args.weights, args.config)` is loading the pre-trained YOLO model from the
# specified weights and configuration files. The `readNet()` function is used to read the model
# architecture and weights into the `net` object, which can then be used for inference.
net = cv2.dnn.readNet(args.weights, args.config)

# creating a 4-dimensional blob from the input image.
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
net.setInput(blob)


def get_output_layers(net):
    """
    The function "get_output_layers" takes a neural network model as input and returns a list of the
    names of the output layers of the model.
    
    :param net: The parameter "net" is expected to be a neural network model object
    :return: a list of output layer names.
    """

    layer_names = net.getLayerNames()
   
    #layer_names[i[0] - 1]
    output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    The function draws a bounding box on an image with the specified class label, confidence score, and
    coordinates.
    
    :param img: The input image on which the bounding box will be drawn
    :param class_id: The class_id parameter represents the class or category of the object detected in
    the image. It is an integer value that corresponds to a specific class label
    :param confidence: The confidence parameter represents the confidence score or probability
    associated with the detected object. It indicates how confident the model is that the object belongs
    to the specified class
    :param x: The x-coordinate of the top-left corner of the bounding box
    :param y: The parameter "y" represents the y-coordinate of the top-left corner of the bounding box
    :param x_plus_w: The x-coordinate of the top-left corner of the bounding box plus the width of the
    bounding box
    :param y_plus_h: The y-coordinate plus the height of the bounding box
    """

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
   

# The line `outs = net.forward(get_output_layers(net))` is performing forward pass inference on the
# neural network model `net` using the `forward()` method. The `forward()` method takes the output
# layer names obtained from the `get_output_layers()` function as input and returns the output of the
# network for the given input image.
outs = net.forward(get_output_layers(net))

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4



detections = np.concatenate(outs, axis=0)
# The line `scores = detections[:, 5:]` is extracting the confidence scores for each detected object
# from the `detections` array.
scores = detections[:, 5:]
# The line `class_ids = np.argmax(scores, axis=1)` is finding the class ID with the highest confidence
# score for each detected object.
class_ids = np.argmax(scores, axis=1)
confidences = scores[np.arange(len(class_ids)), class_ids]
mask = confidences > 0.3
detections = detections[mask]
class_ids = class_ids[mask]
confidences = confidences[mask]

# compute bounding boxes
boxes = detections[:, :4] * np.array([Width, Height, Width, Height])
x = boxes[:, 0] - boxes[:, 2] / 2
y = boxes[:, 1] - boxes[:, 3] / 2
w = boxes[:, 2]
h = boxes[:, 3]

# append results to lists
class_ids = class_ids.tolist()
confidences = confidences.tolist()
boxes = np.column_stack([x, y, w, h]).tolist()

# performing Non-Maximum Suppression (NMS) on the detected bounding boxes.
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# The code block `for i in indices:` is iterating over the indices of the bounding boxes that survived
# the Non-Maximum Suppression (NMS) process.
for i in indices:

    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# display output image    
cv2.imshow("object detection", image)

# wait until any key is pressed
cv2.waitKey()
   
 # save output image to disk
cv2.imwrite("object-detection.jpg", image)

# release resources
cv2.destroyAllWindows()