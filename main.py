import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

random.seed(100)


# The `class_name_to_id_mapping` dictionary is mapping the class names of objects to their
# corresponding class IDs. Each object class is assigned a unique ID, starting from 0. This mapping is
# useful for converting the class names to their corresponding IDs during data preprocessing or model
# training.
class_name_to_id_mapping = {"bicycle": 0,
                           "bus": 1,
                           "car": 2,
                           "cone": 3,
                           "garbage": 4,
                           "human": 5,
                           "moto": 6,
                           "pedestrians": 7,
                           "signs":8,
                           "traffic light": 9,
                           "train": 10,
                           "tree":11,
                           "truck":12                         
                           }

# Get the annotations
# The line is creating a list of file paths for all the annotation files in the "annotations"
# directory.
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]
# is sorting the list of annotation file paths in ascending order. This is done
# to ensure that the annotations are processed in a consistent order, which can be useful for tasks
# such as debugging or reproducibility.
annotations.sort()

# creating a dictionary that maps class IDs to their
# corresponding class names.
class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list):
    """
    The function `plot_bounding_box` takes an image and a list of annotations, and plots bounding boxes
    around the annotated objects on the image.
    
    :param image: The image parameter is the input image on which you want to plot the bounding boxes.
    It should be an instance of the PIL Image class
    :param annotation_list: The annotation_list is a list of annotations for bounding boxes in the
    image. Each annotation is represented as a list with the following elements:
    """
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    # The code snippet is performing a transformation on the annotations of bounding boxes.
    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
   # The code snippet is performing a transformation on the annotations of bounding boxes.
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    # The code snippet is iterating over each transformed annotation in the `transformed_annotations`
    # list. For each annotation, it extracts the object class (`obj_cls`) and the coordinates of the
    # bounding box (`x0`, `y0`, `x1`, `y1`).
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()

# Get any random annotation file 
# The code snippet is randomly selecting an annotation file from the list of annotation files
# (`annotations`). It then opens the selected annotation file and reads its contents. The contents of
# the file are split by newline characters (`\n`) and stored as a list (`annotation_list`). Each
# element in `annotation_list` represents an annotation for a bounding box in the image.
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
# The code snippet is generating the file path for the corresponding image file based on the given
# annotation file path.
image_file = annotation_file.replace("annotations", "images").replace("txt", "jpg")
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)


# Read images and annotations
images = [os.path.join('images', x) for x in os.listdir('images') if x[-3:] == "jpg"]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)


#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    """
    The function `move_files_to_folder` moves a list of files to a specified destination folder, and
    raises an assertion error if any file fails to be moved.
    
    :param list_of_files: A list of file paths that you want to move to the destination folder
    :param destination_folder: The destination_folder parameter is the path to the folder where you want
    to move the files
    """
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
# The code snippet is moving the images and their corresponding annotations into separate folders for
# the train, validation, and test splits of the dataset.
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')

