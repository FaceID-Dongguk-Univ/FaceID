# -----
# author: good-riverdeer
# An implementation of Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
#
# This SRGAN training code is based on tensorlayer's srgan.
# https://github.com/tensorlayer/srgan
# -----

import tensorflow as tf
from glob import glob
import cv2
import time
import datetime
import numpy as np

from networks.resolution.srgan import generator, discriminator
from metrics import PSNR, SSIM


def get_train_data(input_size, target_size, img_list, batch_size):
    def generator_train():
        for img in img_list:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 175.0
            img = img - 1.
            resized = cv2.resize(img, input_size[:2])
            target = cv2.resize(img, target_size[:2])
            yield resized, target

    train_ds = tf.data.Dataset.from_generator(
        generator_train,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(input_size), tf.TensorShape(target_size))
    )
    train_ds = train_ds.shuffle(buffer_size=128)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
    return train_ds


def train(input_size, target_size, img_list, batch_size,
          lr_init, lr_decay,
          n_epoch_init, n_epoch):
    """
    train SRGAN
    :param input_size: LR images tensor shape
    :param target_size: SR images tensor shape
    :param img_list: train images list
    :param batch_size: batch size
    :param lr_init: learning rate - Generator's init_train
    :param lr_decay: learning rate decay ratio
    :param n_epoch_init: number of epochs - Generator's init_train
    :param n_epoch: number of epochs - SRGAN
    """
    decay_every = int(n_epoch / 2)

    # define model
    G = generator(input_size)
    D = discriminator(target_size)
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=target_size)
    vgg.trainable = False
    vgg = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    vgg.trainable = False

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.keras.optimizers.Adam(lr_v)
    g_optimizer = tf.keras.optimizers.Adam(lr_v)
    d_optimizer = tf.keras.optimizers.Adam(lr_v)

    train_ds = get_train_data(input_size, target_size, img_list, batch_size)

    log_dir = "logs\\srgan\\"
    g_init_summary_writer = tf.summary.create_file_writer(
        log_dir + "fit_init\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_patches, hr_patches) in enumerate(train_ds.take(n_step_epoch)):
            if lr_patches.shape[0] != batch_size:
                break
            step_time = time.time()

            with tf.GradientTape() as tape:
                fake_hr_patches = G(lr_patches)
                mse_loss = tf.reduce_mean(tf.math.squared_difference(hr_patches, fake_hr_patches))
                mse_loss = tf.reduce_mean(mse_loss)

            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print(f"""Epoch: [{epoch}/{n_epoch_init}] step: [{step}/{n_step_epoch}], 
                  time: {time.time() - step_time:.3f}s, mse: {mse_loss:.3f}""")

            with g_init_summary_writer.as_default():
                tf.summary.scalar('mse_loss', mse_loss, epoch)

    log_dir = "logs\\srgan\\"
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        for step, (lr_patches, hr_patches) in enumerate(train_ds.take(n_step_epoch)):
            if lr_patches.shape[0] != batch_size:
                break
            step_time = time.time()

            with tf.GradientTape(persistent=True) as tape:
                fake_patches = G(lr_patches)
                logits_fake = D(fake_patches)
                logits_real = D(hr_patches)
                feature_fake = vgg((fake_patches + 1) / 2.)
                feature_real = vgg((hr_patches + 1) / 2.)

                d_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(logits_real), logits_real)
                d_loss1 = tf.reduce_mean(d_loss1)
                d_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(logits_fake), logits_fake)
                d_loss2 = tf.reduce_mean(d_loss2)
                d_loss = d_loss1 + d_loss2

                g_gan_loss = 1e-3 * tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(logits_fake), logits_fake)
                g_gan_loss = tf.reduce_mean(g_gan_loss)

                mse_loss = tf.reduce_mean(tf.math.squared_difference(hr_patches, fake_patches))
                mse_loss = tf.reduce_mean(mse_loss)

                vgg_loss = 2e-6 * tf.reduce_mean(tf.math.squared_difference(feature_real, feature_fake))
                vgg_loss = tf.reduce_mean(vgg_loss)

                g_loss = mse_loss + vgg_loss + g_gan_loss

            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

            psnr_value = np.mean(PSNR(hr_patches, fake_patches))
            ssim_value = np.mean(SSIM(hr_patches, fake_patches))
            print(f"""Epoch: [{epoch}/{n_epoch}] step: [{step}/{n_step_epoch}] time: {time.time() - step_time:.3f}s, 
                  g_loss(mse: {mse_loss:.3f}, vgg:{vgg_loss:.3f}, adv: {g_gan_loss: .3f}) d_loss: {d_loss:.3f}, 
                  Metrics: [PSNR: {psnr_value:.3f}, SSIM: {ssim_value:.3f}]""")

        if epoch != 0 and epoch % decay_every == 0:
            new_lr_decay = lr_decay ** (epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = f" ** new learning rate: {lr_init * new_lr_decay} (for GAN)"
            print(log)

        if epoch != 0 and epoch % 100 == 0:
            G.save_weights(f"./weights/G_{epoch}.h5")
            D.save_weights(f"./weights/D_{epoch}.h5")

            with summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', g_loss, epoch)
                tf.summary.scalar('disc_total_loss', d_loss, epoch)
                tf.summary.scalar('PSNR', psnr_value, epoch)
                tf.summary.scalar('SSIM', ssim_value, epoch)


if __name__ == "__main__":
    pixel = 248
    INPUT_SIZE = (pixel // 4, pixel // 4, 3)
    TARGET_SIZE = (pixel, pixel, 3)
    BATCH_SIZE = 4

    train_img_list = glob("./lfw-deepfunneled/*/*.jpg")

    train(INPUT_SIZE, TARGET_SIZE, train_img_list, BATCH_SIZE,
          lr_init=1e-4, lr_decay=0.1,
          n_epoch_init=1, n_epoch=2000)
