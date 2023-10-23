
import pickle
import csv
import pandas as pd
import numpy as np
import random as rd
import cv2 as cv

from sklearn.model_selection import train_test_split



#ANALISI FILE PKL PER DATSET

# This code snippet is importing the necessary libraries (`os` and `glob`) and setting the current
# working directory (`cd=os.getcwd()`). It then changes the current working directory to the specified
# path (`'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D
# Perception/progetto1/archive/001/annotations/cuboids/'`).

import os
import glob
cd=os.getcwd()
os.chdir('C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/annotations/cuboids/')
image_dir= 'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/camera/front_camera'
files = os.listdir(image_dir)
jpg_files = [f for f in files if f.endswith(".jpg")]

total_df = pd.DataFrame()

# These variables `image_dir1`, `image_dir2`, `image_dir3`, `image_dir4`, `image_dir5`, `image_dir6`
# are storing the file paths of different camera images. Each variable represents the file path of an
# image taken from a specific camera angle. These images are used to associate with the corresponding
# data in the DataFrame.
image_dir1 = 'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/camera/back_camera/00.jpg'
image_dir2 = 'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/camera/front_camera/00.jpg'
image_dir3 = 'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/camera/front_left_camera/00.jpg'
image_dir4 = 'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/camera/front_right_camera/00.jpg'
image_dir5 = 'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/camera/left_camera/00.jpg'
image_dir6 = 'C:/Users/schia/OneDrive/Documenti/UNIVERSITA/Magistrale/2 anno/3D Perception/progetto1/archive/001/camera/right_camera/00.jpg'

# This code snippet is looping through all the files with the extension ".pkl" in the current
# directory. For each file, it opens the file in binary mode (`fh = open(str(file), 'rb')`) and loads
# the data from the file using the `pickle.load()` function (`d = pickle.load(fh)`).
for file in glob.glob("*.pkl"):
    fh = open(str(file), 'rb')
    d = pickle.load(fh)
    df = pd.DataFrame(d)
   
    df.loc[df['camera_used'] == -1, 'image_path'] = image_dir1
    df.loc[df['camera_used'] == 0, 'image_path'] = image_dir2
    df.loc[df['camera_used'] == 1, 'image_path'] = image_dir3
    df.loc[df['camera_used'] == 2, 'image_path'] = image_dir4
    df.loc[df['camera_used'] == 3, 'image_path'] = image_dir5
    df.loc[df['camera_used'] == 4, 'image_path'] = image_dir6
       

    total_df = pd.concat([total_df, df])
   

col_label = total_df['label']

# The code snippet `total_df.dropna(subset=['label'],inplace=True)` is dropping any rows in the
# DataFrame `total_df` where the value in the 'label' column is missing (NaN). The `subset=['label']`
# argument specifies that only the 'label' column should be checked for missing values. The
# `inplace=True` argument means that the changes should be made directly to the DataFrame `total_df`
# without creating a new DataFrame.

total_df.dropna(subset=['label'],inplace=True)
df = total_df.drop(columns=['attributes.object_motion', 'cuboids.sibling_id',
                            'cuboids.sensor_id','attributes.pedestrian_behavior',
                            'attributes.pedestrian_age','attributes.rider_status'], axis=1)

count = df['label'].value_counts()['Pedestrian']

label_str =df['label']

#PROVA DATASET CON SOLO PEDONI
#droppare le colonne che per noi non sono utili
# This code snippet is creating a new DataFrame `front_df` that contains only the data related to
# pedestrians (`label == 'Pedestrian'`) and data from cameras with IDs less than 3 (`camera_used <
# 3`).
df['id'] = range(1, len(df) + 1)
df = df.set_index('id')

ped_df = df.loc[df['label'] == 'Pedestrian']


other_df = df.loc[df['label'] != 'Pedestrian']

other_df1 = other_df.loc[1:count]


result_df = pd.concat([ped_df, other_df1], ignore_index=True)

front_df = result_df.loc[result_df['camera_used'] < 3]

print (front_df.info())





