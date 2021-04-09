import os
from tqdm import tqdm
import tensorflow as tf

from glob import glob
import cv2

from metrics import VGGLoss
from networks.resolution.srgan import Generator, Discriminator, get_srgan

# prepare dataset
train_list = glob("./lfw-deepfunneled/*/*.jpg")
INPUT_SIZE = (250 // 4, 250 // 4, 3)
TARGET_SIZE = (250, 250, 3)
BATCH_SIZE = 32


def get_data():
    for img_path in train_list:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            resized = cv2.resize(img, INPUT_SIZE[:2])
            bicubic = cv2.resize(resized, TARGET_SIZE[:2])
            yield bicubic, img
        except GeneratorExit:
            return


ds = tf.data.Dataset.from_generator(
    get_data,
    output_types=(tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([TARGET_SIZE[0], TARGET_SIZE[1], 3]),
                   tf.TensorShape([TARGET_SIZE[0], TARGET_SIZE[1], 3]))
)

# train-validation split
DATA_SIZE = len(train_list)
TRAIN_SIZE = int(0.7 * DATA_SIZE)
ds = ds.shuffle(buffer_size=1024)
train_ds = ds.take(TRAIN_SIZE)
val_ds = ds.skip(TRAIN_SIZE)

train = train_ds.shuffle(buffer_size=1024).batch(BATCH_SIZE)
val = val_ds.batch(BATCH_SIZE)


# def load_training_data(img_files, ext, number_of_images, train_ratio):
#     number_of_train_images = int(number_of_images * train_ratio)
#
#     files = glob("./lfw-deepfunneled/*/*.jpg")
#
#     x_train = files[:number_of_train_images]
#     x_test = files[number_of_train_images: number_of_images]
#
#     x_train =
def apply_gradient(optimizer, loss_object, model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_object(y, logits)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return logits, loss_value


def train_data_for_one_epoch(optimizer, model, train_metric):
    losses = []
    pbar = tqdm(total=len(list(enumerate(train))), position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
    for step, (x_batch_train, y_batch_train) in enumerate(train):
        logits, loss_value = apply_gradient(optimizer, model, x_batch_train, y_batch_train)

        losses.append(loss_value)

        train_metric(y_batch_train, logits)
        pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
        pbar.update()

    return losses


def perform_validation(loss_object, model, val_metric):
    losses = []
    for x_val, y_val in val:
        val_logits = model(x_val)
        val_loss = loss_object(y_val, val_logits)
        losses.append(val_loss)
        val_metric(y_val, val_logits)
    return losses


generator = Generator(INPUT_SIZE)
# generator.compile(loss=VGGLoss(TARGET_SIZE),
#                   optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
discriminator = Discriminator(TARGET_SIZE)
# discriminator.compile(loss='binary_crossentropy',
#                       optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
srgan = get_srgan(INPUT_SIZE, discriminator, generator)
# srgan.compile(loss=[VGGLoss, 'binary_crossentropy'], loss_weights=[1., 1e-3],
#               optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

epochs_val_losses, epochs_train_losses = [], []

for epoch in range(20):
    print(f"Start of epoch {epoch + 1}")




# def train(epochs, batch_size, dataset, model_save_dir, train_ratio):
#
#     # train-validation split
#     data_size = len(train_list)
#     train_size = int(train_ratio * data_size)
#     dataset = dataset.shuffle(buffer_size=int(0.25 * data_size))
#     train_ds = dataset.take(train_size)
#     val_ds = dataset.skip(train_size)
#
#     loss = VGGLoss(TARGET_SIZE)
#     generator = Generator(INPUT_SIZE)
#     generator.compile(loss=VGGLoss(TARGET_SIZE),
#                       optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
#     discriminator = Discriminator(TARGET_SIZE)
#     discriminator.compile(loss='binary_crossentropy',
#                           optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
#
#     srgan = get_srgan(INPUT_SIZE, discriminator, generator)
#     srgan.compile(loss=[VGGLoss, 'binary_crossentropy'], loss_weights=[1., 1e-3],
#                   optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
#
#     for e in range(1, epochs + 1):
#         print('-' * 15, f'Epoch {e}', '-' * 15)
#         for _ in range(bat)
