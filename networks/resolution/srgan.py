import tensorflow as tf
from metrics import VGGLoss


# class ResBlockGen(tf.keras.Model):
#
#     def __init__(self, filters, kernel_size, strides):
#         super(ResBlockGen, self).__init__()
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#
#         self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.strides, padding='same')
#         self.bn = tf.keras.layers.BatchNormalization(momentum=0.5)
#
#         self.prelu = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
#                                            shared_axes=[1, 2])
#         self.add = tf.keras.layers.Add()
#
#     def call(self, input_tensor):
#         x = self.conv(input_tensor)
#         x = self.bn(x)
#         x = self.prelu(x)
#
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.add([input_tensor, x])
#         return x


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


# class UpSamplingBlock(tf.keras.Model):
#
#     def __init__(self, filters, kernel_size, strides):
#         super(UpSamplingBlock, self).__init__()
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#
#         self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.strides, padding='same')
#         self.up_sampling_2d = tf.keras.layers.UpSampling2D(size=2)
#         self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
#
#     def call(self, input_tensor):
#         x = self.conv(input_tensor)
#         x = self.up_sampling_2d(x)
#         x = self.leaky_relu(x)
#         return x


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


# class Generator(tf.keras.Model):
#
#     def __init__(self, noise_shape):
#         super(Generator, self).__init__()
#         self.gen_input = tf.keras.layers.Input(shape=noise_shape)
#
#         self.conv1 = tf.keras.layers.Conv2D(64, 9, 1, padding='same')
#         self.conv2 = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
#         self.conv3 = tf.keras.layers.Conv2D(3, 9, 1, padding='same')
#
#         self.res_block_gen = ResBlockGen(64, 3, 1)
#         self.up_sampling_block = UpSamplingBlock(256, 3, 1)
#
#         self.prelu = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None,
#                                            alpha_constraint=None, shared_axes=[1, 2])
#         self.add = tf.keras.layers.Add()
#         self.bn = tf.keras.layers.BatchNormalization(momentum=0.5)
#         self.tanh = tf.keras.layers.Activation('tanh')
#
#     def call(self, input_tensor):
#         gen_input = self.gen_input(input_tensor)
#         x = self.conv1(gen_input)
#         gen_x = self.prelu(x)
#
#         # Using 16 Residual Blocks
#         for i in range(16):
#             x = self.res_block_gen(x)
#
#         x = self.conv2(x)
#         x = self.bn(x)
#         x = self.add([gen_x, x])
#
#         #Using 2 UpSampling Blocks
#         for i in range(2):
#             x = self.up_sampling_block(x)
#
#         x = self.conv3(x)
#         x = self.tanh(x)
#         return tf.keras.Model(gen_input, x)


# class DiscriminatorBlock(tf.keras.Model):
#
#     def __init__(self, filters, kernel_size, strides):
#         super(DiscriminatorBlock, self).__init__()
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#
#         self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.strides, padding='same')
#         self.bn = tf.keras.layers.BatchNormalization(momentum=0.5)
#         self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
#
#     def call(self, input_tensor):
#         x = self.conv(input_tensor)
#         x = self.bn(x)
#         x = self.leaky_relu(x)
#         return x
#
#
# class Discriminator(tf.keras.Model):
#
#     def __init__(self, target_shape):
#         super(Discriminator, self).__init__()
#         self.dis_input = tf.keras.layers.Input(shape=target_shape)
#
#         self.conv = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
#         self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
#
#         self.db1 = DiscriminatorBlock(64, 3, 2)
#         self.db2 = DiscriminatorBlock(128, 3, 1)
#         self.db3 = DiscriminatorBlock(128, 3, 2)
#         self.db4 = DiscriminatorBlock(256, 3, 1)
#         self.db5 = DiscriminatorBlock(256, 3, 2)
#         self.db6 = DiscriminatorBlock(512, 3, 1)
#         self.db7 = DiscriminatorBlock(512, 3, 2)
#
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(1024)
#         self.classifier = tf.keras.layers.Dense(1)
#         self.sigmoid = tf.keras.layers.Activation('sigmoid')
#
#     def call(self, input_tensor):
#         dis_input = self.dis_input(input_tensor)
#         x = self.conv(dis_input)
#         x = self.leaky_relu(x)
#
#         x = self.db1(x)
#         x = self.db2(x)
#         x = self.db3(x)
#         x = self.db4(x)
#         x = self.db5(x)
#         x = self.db6(x)
#         x = self.db7(x)
#
#         x = self.flatten(x)
#         x = self.dense(x)
#         x = self.leaky_relu(x)
#
#         x = self.classifier(x)
#         x = self.sigmoid(x)
#         return tf.keras.Model(dis_input, x)


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

    def generator(self):

        gen_input = tf.keras.layers.Input(shape=self.noise_shape)

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


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):
        dis_input = tf.keras.layers.Input(shape=self.image_shape)

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


def get_srgan(shape, discriminator, generator):
    discriminator.trainable = False
    srgan_input = tf.keras.layers.Input(shape=shape)
    x = generator(srgan_input)
    srgan_output = discriminator(x)

    srgan = tf.keras.Model(inputs=srgan_input, outputs=[x, srgan_output])
    return srgan
