# -----
# author: good-riverdeer
# An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition
# https://arxiv.org/abs/1801.07698
#
# This ArcFace code is based on 4uiiurz1's keras-arcface.
# https://github.com/4uiiurz1/keras-arcface
# -----

import tensorflow as tf
from tensorflow.keras.applications import ResNet50

from metrics import ArcFace


def resnet50_arcface(img_size, num_classes):
    input_tensor = tf.keras.layers.Input(shape=img_size)
    input_label = tf.keras.layers.Input(shape=(num_classes, ))

    resnet = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
    model = tf.keras.layers.GlobalAveragePooling2D()(resnet.output)
    model = tf.keras.layers.Dense(512, kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(model)
    model = ArcFace(num_classes, regularizer=tf.keras.regularizers.l2(1e-4))([model, input_label])
    return tf.keras.Model(inputs=[input_tensor, input_label], outputs=[model])
