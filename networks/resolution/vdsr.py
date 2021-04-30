"""
author: good-riverdeer
An implementation of VDSR (Super-resolution using very deep convolutional networks)
https://openaccess.thecvf.com/content_cvpr_2016/html/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.html

This VDSR code is based on GeorgeSeif's VDSR-Keras.
https://github.com/GeorgeSeif/VDSR-Keras
"""
import tensorflow as tf


def vdsr(img_size, filters):
    input_img = tf.keras.layers.Input(shape=img_size)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(input_img)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(1, 3, padding='same', kernel_initializer='he_normal')(x)
    output = tf.keras.layers.Add()([x, input_img])

    return tf.keras.Model(input_img, output)
