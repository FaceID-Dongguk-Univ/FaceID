# -----
# author: good-riverdeer
# An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition
# https://arxiv.org/abs/1801.07698
#
# This ArcFace code is based on 4uiiurz1's keras-arcface.
# https://github.com/4uiiurz1/keras-arcface
# -----
import os
import tensorflow as tf
import numpy as np

from metrics import PSNR, SSIM
from data.dataloader_srgan import DataLoader
from networks.resolution.srgan import FastSRGAN
from utils import load_yaml


@tf.function
def pretrain_step(model, x, y):
    """
    Single step of generator pre-training.
    Args:
        model: A model object with a tf keras compiled generator.
        x: The low resolution image tensor.
        y: The high resolution image tensor.
    """
    with tf.GradientTape() as tape:
        fake_hr = model.generator(x)
        loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)

    grads = tape.gradient(loss_mse, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(grads, model.generator.trainable_variables))

    return loss_mse


def pretrain_generator(model, dataset, writer):
    """Function that pretrains the generator slightly, to avoid local minima.
    Args:
        model: The keras model to train.
        dataset: A tf dataset object of low and high res images to pretrain over.
        writer: A summary writer object.
    Returns:
        None
    """
    with writer.as_default():
        iteration = 0
        for _ in range(1):
            for x, y in dataset:
                loss = pretrain_step(model, x, y)
                if iteration % 20 == 0:
                    print(f"pretrain generator: iterations: {iteration} | MSE Loss: {loss:.4f}")
                    tf.summary.scalar('MSE Loss', loss, step=tf.cast(iteration, tf.int64))
                    writer.flush()
                iteration += 1


@tf.function
def train_step(model, x, y):
    """Single train step function for the SRGAN.
    Args:
        model: An object that contains a tf keras compiled discriminator model.
        x: The low resolution input image.
        y: The desired high resolution output image.

    Returns:
        d_loss: The mean loss of the discriminator.
    """
    # Label smoothing for better gradient flow
    valid = tf.ones((x.shape[0],) + model.disc_patch)
    fake = tf.zeros((x.shape[0],) + model.disc_patch)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # From low res. image generate high res. version
        fake_hr = model.generator(x)

        # Train the discriminators (original images = real / generated = Fake)
        valid_prediction = model.discriminator(y)
        fake_prediction = model.discriminator(fake_hr)

        # Generator loss
        content_loss = model.content_loss(y, fake_hr)
        adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
        mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
        perceptual_loss = content_loss + adv_loss + mse_loss

        # Discriminator loss
        valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
        fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
        d_loss = tf.add(valid_loss, fake_loss)

    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

    return d_loss, adv_loss, content_loss, mse_loss


def train(model, dataset, log_iter, writer, weight_dir):
    """
    Function that defines a single training step for the SR-GAN.
    Args:
        model: An object that contains tf keras compiled generator and
               discriminator models.
        dataset: A tf data object that contains low and high res images.
        log_iter: Number of iterations after which to add logs in
                  tensorboard.
        writer: Summary writer
        weight_dir: directory to save checkpoints
    """
    with writer.as_default():
        # Iterate over dataset
        for x, y in dataset:
            disc_loss, adv_loss, content_loss, mse_loss = train_step(model, x, y)
            # Log tensorboard summaries if log iteration is reached.
            if model.iterations % log_iter == 0:
                print(f"SRGAN training | step: {model.iterations} | adv_loss: {adv_loss:.4f} | content_loss: {content_loss:.4f} | " +
                      f"mse_loss: {mse_loss:.4f} | disc_loss: {disc_loss:.4f}")
                tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
                tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=model.iterations)
                tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
                tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                # model.generator.save(args['weight_dir'] + f'/generator.h5')
                # model.discriminator.save(args['weight_dir'] + f'/discriminator.h5')
                model.generator.save_weights(weight_dir + f'/srgan/generator_{model.iterations}.ckpt')
                model.discriminator.save_weights(weight_dir + f'/srgan/discriminator_{model.iterations}.ckpt')
                writer.flush()
            model.iterations += 1


@tf.function
def validation_step(model, x, y):
    valid = tf.ones((x.shape[0],) + model.disc_patch)
    fake = tf.zeros((x.shape[0],) + model.disc_patch)

    fake_hr = model.generator(x)

    valid_prediction = model.discriminator(y)
    fake_prediction = model.discriminator(fake_hr)

    content_loss = model.content_loss(y, fake_hr)
    adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
    mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)

    psnr = PSNR(y, fake_hr)
    ssim = SSIM(y, fake_hr)
    psnr = tf.reduce_mean(psnr)
    ssim = tf.reduce_mean(ssim)

    valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
    d_loss = tf.add(valid_loss, fake_loss)

    return adv_loss, content_loss, mse_loss, d_loss, psnr, ssim


def validation(model, dataset, writer):
    adv_losses = []
    content_losses = []
    mse_losses = []
    disc_losses = []
    psnrs = []
    ssims = []

    for x, y in dataset:
        adv_loss, content_loss, mse_loss, disc_loss, psnr, ssim = validation_step(model, x, y)
        adv_losses.append(adv_loss)
        content_losses.append(content_loss)
        mse_losses.append(mse_loss)
        disc_losses.append(disc_loss)
        psnrs.append(psnr)
        ssims.append(ssim)

    adv_loss = np.array(adv_losses).mean()
    content_loss = np.array(content_losses).mean()
    mse_loss = np.array(mse_losses).mean()
    disc_loss = np.array(disc_losses).mean()
    psnr = np.array(psnrs).mean()
    ssim = np.array(ssims).mean()

    print(f"SRGAN evaluating | adv_loss: {adv_loss:.4f} | content_loss: {content_loss:.4f} | " +
          f"mse_loss: {mse_loss:.4f} | disc_loss: {disc_loss:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}")

    with writer.as_default():
        tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
        tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
        tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
        tf.summary.scalar('Discriminator Loss', disc_loss, step=model.iterations)
        tf.summary.scalar('PSNR', psnr, step=model.iterations)
        tf.summary.scalar('SSIM', ssim, step=model.iterations)
        tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
        tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
        tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                         step=model.iterations)
        writer.flush()


def main():
    cfg = load_yaml('configs/srgan.yaml')

    # create directory for saving trained models and logging.
    if not os.path.exists(cfg['weight_dir']):
        os.makedirs(cfg['weight_dir'])
    if not os.path.exists(cfg['log_dir']):
        os.makedirs(cfg['log_dir'])

    # Create the tensorflow dataset.
    train_ds = DataLoader(cfg['train_image_dir'], cfg['hr_size']).dataset(cfg['batch_size'])
    val_ds = DataLoader(cfg['val_image_dir'], cfg['hr_size']).dataset(cfg['batch_size'])

    # Initialize the GAN object.
    gan = FastSRGAN(cfg)

    # Define the directory for saving pretrainig loss tensorboard summary.
    model_name = f"srgan-lr{cfg['lr']}-e{cfg['epochs']}-bs{cfg['batch_size']}"
    pretrain_summary_writer = tf.summary.create_file_writer(cfg['log_dir'] + '/' + model_name + '/pretrain')

    # Run pre-training.
    pretrain_generator(gan, train_ds, pretrain_summary_writer)

    # Define the directory for saving the SRGAN training tensorbaord summary.
    train_summary_writer = tf.summary.create_file_writer(cfg['log_dir'] + '/' + model_name + '/train')
    val_summary_writer = tf.summary.create_file_writer(cfg['log_dir'] + '/' + model_name + '/validation')

    # Run training.
    for i in range(cfg['epochs']):
        print(f"EPOCH: {i + 1}")
        train(gan, train_ds, cfg['save_iter'], train_summary_writer, cfg['weight_dir'])
        validation(gan, val_ds, val_summary_writer)


if __name__ == '__main__':
    main()
