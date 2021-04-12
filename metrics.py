# -----
# author: good-riverdeer
# An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition
# https://arxiv.org/abs/1801.07698
#
# This ArcFace code is based on 4uiiurz1's keras-arcface.
# https://github.com/4uiiurz1/keras-arcface
# This PSNR, SSIM code is based on hieubkset's Keras-Image-Super-Resolution.
# https://github.com/hieubkset/Keras-Image-Super-Resolution
# -----

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers

import tensorflow as tf


# def tf_log10(x):
#     numerator = tf.math.log(x)
#     denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
#     return numerator / denominator
#
#
# def PSNR(y_true, y_pred):
#     max_pixel = 1.0
#     return 10.0 * tf_log10((max_pixel**2) / (K.mean(K.square(y_pred - y_true))))
#
#
# def SSIM(y_true, y_pred):
#     max_pixel = 1.0
#     return tf.image.ssim(y_pred, y_true, max_pixel)


def rgb_to_y(image):
    image = tf.image.rgb_to_yuv(image)
    image = (image * (235 - 16) + 16) / 255.0
    return image[:, :, :, 0]


def crop(image):
    margin = 4
    image = image[:, margin:-margin, margin:-margin]
    return tf.expand_dims(image, -1)


def un_normalize(hr, sr):
    hr = hr * 0.5 + 0.5
    sr = tf.clip_by_value(sr, -1, 1)
    sr = sr * 0.5 + 0.5
    return hr, sr


def PSNR(hr, sr):
    hr, sr = un_normalize(hr, sr)
    hr = rgb_to_y(hr)
    sr = rgb_to_y(sr)
    hr = crop(hr)
    sr = crop(sr)
    return tf.image.psnr(hr, sr, max_val=1.0)


def SSIM(hr, sr):
    hr, sr = un_normalize(hr, sr)
    hr = rgb_to_y(hr)
    sr = rgb_to_y(sr)
    hr = crop(hr)
    sr = crop(sr)
    return tf.image.ssim(hr, sr, max_val=1.0)


class VGGLoss(tf.keras.losses.Loss):

    def __init__(self, input_shape):
        super(VGGLoss, self).__init__()
        self.input_shape = input_shape

    def call(self, y_true, y_pred):

        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)
        vgg19.trainable = False

        for l in vgg19.layers:
            l.trainable = False

        model = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
        return K.mean(K.square(model(y_true) - model(y_pred)))


class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def get_config(self):
        base_config = super(ArcFace, self).get_config()
        base_config['n_classes'] = self.n_classes
        base_config["s"] = self.s
        base_config["m"] = self.m
        base_config['regularizer'] = self.regularizer
        return base_config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]

        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)

        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)

        # dot product
        logits = x @ W

        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)

        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y

        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class SphereFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(self.m * theta)
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class CosFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
