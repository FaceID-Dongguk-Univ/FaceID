# -----
# author: good-riverdeer
# An implementation of Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
#
# This SRGAN code is based on deepak112's Keras-SRGAN.
# https://github.com/deepak112/Keras-SRGAN
# -----

import tensorflow as tf


def res_block_gen(model, kernel_size, filters, strides):
    gen = model
    model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
    model = tf.keras.layers.BatchNormalization(momentum=0.5)(model)
    model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                  shared_axes=[1, 2])(model)
    model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
    model = tf.keras.layers.BatchNormalization(momentum=0.5)(model)

    model = tf.keras.layers.Add()([gen, model])

    return model


def up_sampling_block(model, kernel_size, filters, strides):
    model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
    model = tf.keras.layers.UpSampling2D(size=2)(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    return model


def discriminator_block(model, filters, kernel_size, strides):
    model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
    model = tf.keras.layers.BatchNormalization(momentum=0.5)(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    return model


def generator(input_shape=(100, 100, 3)):
    gen_input = tf.keras.layers.Input(shape=input_shape)

    model = tf.keras.layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
    model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                  shared_axes=[1, 2])(model)
    gen_model = model

    # Using 16 Residual Blocks
    for index in range(16):
        model = res_block_gen(model, 3, 64, 1)
    model = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    model = tf.keras.layers.BatchNormalization(momentum=0.5)(model)
    model = tf.keras.layers.add([gen_model, model])

    # Using 2 UpSampling Blocks
    for index in range(2):
        model = up_sampling_block(model, 3, 256, 1)
    model = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
    model = tf.keras.layers.Activation('tanh')(model)
    generator_model = tf.keras.Model(inputs=gen_input, outputs=model)

    return generator_model


def discriminator(image_shape=(250, 250, 3)):
    dis_input = tf.keras.layers.Input(shape=image_shape)

    model = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    model = discriminator_block(model, 64, 3, 2)
    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 128, 3, 2)
    model = discriminator_block(model, 256, 3, 1)
    model = discriminator_block(model, 256, 3, 2)
    model = discriminator_block(model, 512, 3, 1)
    model = discriminator_block(model, 512, 3, 2)

    model = tf.keras.layers.Flatten()(model)
    model = tf.keras.layers.Dense(1024)(model)
    model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)

    model = tf.keras.layers.Dense(1)(model)
    model = tf.keras.layers.Activation('sigmoid')(model)

    discriminator_model = tf.keras.Model(inputs=dis_input, outputs=model)

    return discriminator_model
