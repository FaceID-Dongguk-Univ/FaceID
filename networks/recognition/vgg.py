import tensorflow as tf
from tensorflow.keras import regularizers
from metrics import ArcFace


class Block(tf.keras.Model):

    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions

        for i in range(repetitions):
            vars(self)[f"conv2D_{i}"] = tf.keras.layers.Conv2D(self.filters, self.kernel_size,
                                                               activation='relu', padding='same')

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(pool_size, pool_size), strides=(strides, strides))

    def call(self, inputs):
        conv2D_0 = vars(self)['conv2D_0']
        x = conv2D_0(inputs)

        for i in range(1, self.repetitions):
            conv2D_i = vars(self)[f"conv2D_{i}"]
            x = conv2D_i(x)

        max_pool = vars(self)['max_pool'](x)

        return max_pool


class VGG16(tf.keras.Model):

    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.weight_decay = 1e-4

        self.y = tf.keras.layers.Input(shape=(num_classes, ))

        self.block_a = Block(filters=64, kernel_size=3, repetitions=2)
        self.block_b = Block(filters=128, kernel_size=3, repetitions=2)
        self.block_c = Block(filters=256, kernel_size=3, repetitions=3)
        # self.block_d = Block(filters=512, kernel_size=3, repetitions=3)
        # self.block_e = Block(filters=512, kernel_size=3, repetitions=3)

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal',
                                        kernel_regularizer=regularizers.l2(self.weight_decay))
        self.classifier = ArcFace(num_classes, regularizer=regularizers.l2(self.weight_decay))

    def call(self, inputs):
        x, y = inputs
        # y = self.y

        x = self.block_a(x)
        x = self.block_b(x)
        x = self.block_c(x)
        # x = self.block_d(x)
        # x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier([x, y])
        return x
