import cv2
import os
import random
import numpy as np

xs = []
ys = []

# Points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# Base path for dataset
base_path = '../../data/driving_dataset/'

# Reading data
with open(os.path.join(base_path, 'data.txt')) as f:
    for line in f:
        image_path = os.path.join(base_path, line.split()[0])
        xs.append(image_path)
        '''
        The paper by NVIDIA uses the inverse of turning radius, 
        but the steering wheel angle is proportional to the inverse of turning radius.
        So, the steering wheel angle in the output is used as radians (pi/180).
        '''
        ys.append(float(line.split()[1]) * 3.14159265 / 180)

num_images = len(xs)

# Shuffling list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(ys) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []

    for i in range(batch_size):
        image_path = train_xs[(train_batch_pointer + i) % num_train_images]
        img = cv2.imread(image_path)

        if img is None:
            print(f"Skipping missing image: {image_path}")
            continue  # Skip this image

        img = img[-150:]  # Crop the last 150 pixels
        img = cv2.resize(img, (200, 66)) / 255.0
        x_out.append(img)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])

    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []

    for i in range(batch_size):
        image_path = val_xs[(val_batch_pointer + i) % num_val_images]
        img = cv2.imread(image_path)

        if img is None:
            print(f"Skipping missing validation image: {image_path}")
            continue  # Skip this image

        img = img[-150:]
        img = cv2.resize(img, (200, 66)) / 255.0
        x_out.append(img)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])

    val_batch_pointer += batch_size
    return x_out, y_out
