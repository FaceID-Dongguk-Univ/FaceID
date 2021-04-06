import tensorflow as tf
from metrics import VGGLoss


class ResBlockGen(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides):
        super(ResBlockGen, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.strides, padding='same')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.5)

        self.prelu = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                           shared_axes=[1, 2])
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.prelu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.add([input_tensor, x])
        return x


# def res_block_gen(model, kernel_size, filters, strides):
#     gen = model
#     model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
#     model = tf.keras.layers.BatchNormalization(momentum=0.5)(model)
#     model = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
#                                   shared_axes=[1, 2])(model)
#     model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
#     model = tf.keras.layers.BatchNormalization(momentum=0.5)(model)
#
#     model = tf.keras.layers.Add()([gen, model])
#
#     return model


class UpSamplingBlock(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides):
        super(UpSamplingBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.strides, padding='same')
        self.up_sampling_2d = tf.keras.layers.UpSampling2D(size=2)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.up_sampling_2d(x)
        x = self.leaky_relu(x)
        return x


# def up_sampling_block(model, kernel_size, filters, strides):
#     model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
#     model = tf.keras.layers.UpSampling2D(size=2)(model)
#     model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
#
#     return model


# def discriminator_block(model, filters, kernel_size, strides):
#     model = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(model)
#     model = tf.keras.layers.BatchNormalization(momentum=0.5)(model)
#     model = tf.keras.layers.LeakyReLU(alpha=0.2)(model)
#
#     return model


class Generator(tf.keras.Model):

    def __init__(self, noise_shape):
        super(Generator, self).__init__()
        self.noise_shape = noise_shape
        self.gen_input = tf.keras.layers.Input(shape=self.noise_shape)

        self.conv1 = tf.keras.layers.Conv2D(64, 9, 1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(3, 9, 1, padding='same')

        self.res_block_gen = ResBlockGen(64, 3, 1)
        self.up_sampling_block = UpSamplingBlock(256, 3, 1)

        self.prelu = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None,
                                           alpha_constraint=None, shared_axes=[1,2])
        self.add = tf.keras.layers.Add()
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, input_tensor):
        x = self.gen_input(input_tensor)
        x = self.conv1(x)
        gen_x = self.prelu(x)

        # Using 16 Residual Blocks
        for i in range(16):
            x = self.res_block_gen(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.add([gen_x, x])

        #Using 2 UpSampling Blocks
        for i in range(2):
            x = self.up_sampling_block(x)

        x = self.conv3(x)
        x = self.tanh(x)
        return x


class DiscriminatorBlock(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides):
        super(DiscriminatorBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, self.strides, padding='same')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        # self.input_shape = input_shape

        self.conv = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.db1 = DiscriminatorBlock(64, 3, 2)
        self.db2 = DiscriminatorBlock(128, 3, 1)
        self.db3 = DiscriminatorBlock(128, 3, 2)
        self.db4 = DiscriminatorBlock(256, 3, 1)
        self.db5 = DiscriminatorBlock(256, 3, 2)
        self.db6 = DiscriminatorBlock(512, 3, 1)
        self.db7 = DiscriminatorBlock(512, 3, 2)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1024)
        self.classifier = tf.keras.layers.Dense(1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.leaky_relu(x)

        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.db4(x)
        x = self.db5(x)
        x = self.db6(x)
        x = self.db7(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.leaky_relu(x)

        x = self.classifier(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    input_shape = (41, 41, 3)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    generator = Generator(input_shape)
    generator.compile(loss=VGGLoss(input_shape), optimizer=optimizer)

    discriminator = Discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    srgan = tf.keras.Model(inputs=generator.input, outputs=[generator, discriminator])
    srgan.compile(loss=[VGGLoss(input_shape), 'binary_crossentropy'],
                  loss_weights=[1., 1e-3],
                  optimizer=optimizer)


    def train_srgan(srgan, dataset, epochs=50):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for real_images in dataset:
                batch_size = real_images.shape[0]







