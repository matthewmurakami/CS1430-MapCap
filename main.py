
#Import Libraries
import os
import sys
import argparse
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import datasets,models,layers
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def main():
    """ Main function. """
    # (D0, D1), D_info = tfds.load(
    # "downsampled_imagenet", as_supervised=True, split=["train[:80%]", "test"], with_info=True)

    # X0, X1 = [np.array([r[0] for r in tfds.as_numpy(D)]) for D in (D0, D1)]
    # Y0, Y1 = [np.array([r[1] for r in tfds.as_numpy(D)]) for D in (D0, D1)]




if __name__ == "__main__":
    # Make arguments global
    ARGS = parse_args()

    main()
