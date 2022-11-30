
#Import Libraries
import os
import sys
import argparse
import re
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

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
        weights = None, 
        input_shape = (256, 256, 3),
        classes = 200 )
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()

    model.fit(train_ds, epochs=10, validation_data=validation_ds)



    #Load the image to run the model on
    image = keras.preprocessing.image.load_img('cat_front.jpeg',target_size=(224,224))
    plt.imshow(image)
    plt.show()

    #Preprocess image to get it into the right format for the model
    img = keras.preprocessing.image.img_to_array(image)
    img = img.reshape((1, *img.shape))
    preds = model.predict(img)
    print('Predicted:', keras.applications.resnet50.decode_predictions(preds, top=3)[0])
    
    #Calculate gradient to see which pixels conribute the most
    images = tf.Variable(img, dtype=float)
    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
    
    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)

    #Get Saliency map
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

    #Normalize gradient between 0 and 1
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    #Output
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(image)
    i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
    fig.colorbar(i)




if __name__ == "__main__":
    ARGS = parse_args()
    main()
