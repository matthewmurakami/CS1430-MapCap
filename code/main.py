
#Import Libraries
import os
import sys
import argparse
import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from setup import setup

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Flag to perform setup')
    
    return parser.parse_args()

def main():
    """ Main function. """

    if ARGS.setup:
        setup(ARGS.data)


    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=ARGS.data+'train',
        labels='inferred',
        label_mode='categorical',
        batch_size=100,
        image_size=(256, 256))
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=ARGS.data+'val/images',
        labels='inferred',
        label_mode='categorical',
        batch_size=100,
        image_size=(256, 256))

    model = tf.keras.applications.ResNet50(
        weights=None, input_shape=(256, 256, 3), classes=200)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(train_ds, epochs=10, validation_data=validation_ds)



if __name__ == "__main__":
    ARGS = parse_args()
    main()
