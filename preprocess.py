import os
import tensorflow as tf


def img_preprocess(image_path):
    read_img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(read_img)
    img = tf.image.resize(img, (105,105))
    img = img / 255.0 #scales image in between 0-1
    return img