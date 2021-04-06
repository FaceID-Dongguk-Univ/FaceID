import tensorflow as tf
from tensorflow.keras.datasets import mnist
from networks.recognition.resnet import ResNet
from tensorflow.keras.applications import ResNet50
from metrics import ArcFace
import numpy as np

# (X, y), (X_test, y_test) = mnist.load_data()
#
# X = X[:, :, :, np.newaxis].astype('float32') / 255
# X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255
#
# y = tf.keras.utils.to_categorical(y, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)


# model = VGG16(10)
# model = ResNet(10)
model = ResNet50(include_top=False, weights=None)
model = tf.keras.layers.Dense(n_classes)(model)
model = ArcFace(n_classes, regularizer=tf.keras.regularizers.l2(0.01))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
#
# model.fit([X, y], y, validation_data=([X_test, y_test], y_test),
#           epochs=5)

model.summary()
