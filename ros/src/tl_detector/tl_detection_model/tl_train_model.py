#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import array_to_img, img_to_array
from keras import backend as K
import os
import shutil


def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 0:
        shutil.rmtree(testing_data_dir, ignore_errors=True)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 0:
        shutil.rmtree(training_data_dir, ignore_errors=True)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete training data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training images.")
    print("Processed " + str(num_testing_files) + " testing images.")

    return num_training_files, num_testing_files

#########################################################################################

if __name__ == '__main__':
  
  scene_type = 'site'

  # Learning parameters
  batch_size = 32
  epochs = 100

  # split the data images into training and test
  all_data_dir = "tl_data/%s_data" % scene_type 
  training_data_dir="tl_data/train"
  testing_data_dir="tl_data/test" 

  split_dataset_into_test_and_train_sets(
      all_data_dir=all_data_dir, 
      training_data_dir=training_data_dir, 
      testing_data_dir=testing_data_dir, 
      testing_data_pct=0.2)

  num_train_samples = sum([len(files) for r, d, files in os.walk(training_data_dir)])
  num_test_samples = sum([len(files) for r, d, files in os.walk(testing_data_dir)])

  # First check that we are usinga  GPU
  from tensorflow.python.client import device_lib
  print(device_lib.list_local_devices())

  # Define the simple convolutional model
  img_width, img_height = 224, 224
  input_shape = (img_width, img_height, 3)

  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4))
  model.add(Activation('softmax'))

  model.compile(
      optimizer='adagrad', 
      loss='categorical_crossentropy',
      metrics=['accuracy'])

  # Define the data generators with augmentation
  train_datagen = ImageDataGenerator(
      data_format='channels_last',
      rescale=1./255,
      rotation_range=10,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)

  train_generator = train_datagen.flow_from_directory(
      directory=training_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode='categorical')

  test_generator = train_datagen.flow_from_directory(
      directory=testing_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode='categorical')

  print(test_generator.class_indices)

  # Unleash the training
  model.fit_generator(
      generator=train_generator,
      steps_per_epoch=int(num_train_samples/batch_size),
      epochs=epochs,
      validation_data=test_generator,
      validation_steps=int(num_test_samples/batch_size),
      use_multiprocessing=False)

  # Save the model
  print(test_generator.class_indices)
  model_name = 'TLD_%s.h5' % scene_type
  model.save(model_name) 


