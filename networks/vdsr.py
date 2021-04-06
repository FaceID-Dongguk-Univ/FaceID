import tensorflow as tf
from metrics import PSNR


class ConvBlock(tf.keras.Model):

    def __init__(self, filters):
        super(ConvBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')
        self.act = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.act(x)

        x = self.conv(x)
        x = self.act(x)

        x = self.conv(x)
        x = self.act(x)

        x = self.conv(x)
        x = self.act(x)

        return self.add([x, input_tensor])


class VDSR(tf.keras.Model):

    def __init__(self, img_size, filters):
        super(VDSR, self).__init__()
        self.img_size = img_size

        self.conv_block = ConvBlock(filters)
        self.conv_1 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')
        self.conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv_block(input_tensor)
        x = self.conv(x)

        x = self.conv_block(x)
        x = self.conv(x)

        x = self.conv_block(x)
        x = self.conv(x)

        x = self.conv_block(x)
        x = self.conv_1(x)
        return x


if __name__ == '__main__':
    model = VDSR((41, 41, 1), 64)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                  metrics=[PSNR, 'acc'])
    # model.summary()
