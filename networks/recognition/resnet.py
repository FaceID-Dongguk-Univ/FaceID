import tensorflow as tf
from tensorflow.keras import regularizers
from metrics import ArcFace


class IdentityBlock(tf.keras.Model):

    def __init__(self, filters, kernel_size, stride):
        super(IdentityBlock, self).__init__(name='')

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, 1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)
        return x


class ResNet(tf.keras.Model):

    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.weight_decay = 1e-4

        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3))

        self.id1a = IdentityBlock(64, 3, 1)
        self.id1b = IdentityBlock(64, 3, 1)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = ArcFace(num_classes, regularizer=regularizers.l2(self.weight_decay))

    def call(self, inputs):
        x, y = inputs

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)
        return self.classifier([x, y])
